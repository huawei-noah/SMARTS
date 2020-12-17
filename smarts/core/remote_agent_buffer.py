# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
import logging
import atexit
import time
import random
import grpc

from multiprocessing import Process
from multiprocessing.connection import Client
from concurrent import futures

from smarts.zoo import master as zoo_master
from smarts.zoo import agent_pb2
from smarts.zoo import agent_pb2_grpc
from .remote_agent import RemoteAgent, RemoteAgentException
from .utils.networking import find_free_port


class RemoteAgentBuffer:
    def __init__(self, zoo_master_addrs=None, auth_key=None, buffer_size=3):
        """
        Args:
            zoo_master_addrs:
                List of (ip, port) tuples of Zoo Masters, used to instantiate remote social agents
            auth_key:
                Authentication key of type string for communication with Zoo Masters
            buffer_size: 
                Number of RemoteAgents to pre-initialize and keep running in the background, must be non-zero (default: 3)
        """
        assert buffer_size > 0

        self._log = logging.getLogger(self.__class__.__name__)

        self._local_zoo_master = None
        self._local_zoo_master_addr = None

        if zoo_master_addrs is None:
            # The user has not specified any remote zoo masters.
            # We need to spawn a local zoo master.
            self._local_zoo_master_addr = self._spawn_local_zoo_master()
            zoo_master_addrs = [self._local_zoo_master_addr]

        self._zoo_master_addrs = zoo_master_addrs
        self._buffer_size = buffer_size
        self._replenish_threadpool = futures.ThreadPoolExecutor()
        self._agent_buffer = [self._remote_agent_future() for _ in range(buffer_size)]

        atexit.register(self.destroy)

    def destroy(self):
        if atexit:
            atexit.unregister(self.destroy)

        for remote_agent_future in self._agent_buffer:
            try:
                remote_agent = remote_agent_future.result()
                remote_agent.terminate()
            except Exception as e:
                self._log.error(
                    f"Exception while tearing down buffered remote agent: {repr(e)}"
                )

        # Note: important to teardown the local zoo master after we purge the remote agents
        #       since they may still require the local zoo master to for instantiation.
        if self._local_zoo_master is not None:
            self._local_zoo_master.kill()
            self._local_zoo_master.join()

    def _spawn_local_zoo_master(self, retries=3):
        port = find_free_port()

        self._local_zoo_master = Process(target=zoo_master.serve, args=(port,))
        self._local_zoo_master.start()
        local_address = ("localhost", port)
        return local_address

    def _build_remote_agent(self, zoo_master_addrs):
        # Get a random zoo master
        zoo_master_addr = random.choice(zoo_master_addrs)

        # Connect to the random zoo master
        ip, port = zoo_master_addr
        channel = grpc.insecure_channel(f"{ip}:{port}")
        try:
            # Wait until the grpc server is ready or timeout after 30 seconds
            grpc.channel_ready_future(channel).result(timeout=30)
        except grpc.FutureTimeoutError:
            raise RemoteAgentException(
                "Timeout error in connecting to remote zoo server."
            ) from e
        stub = agent_pb2_grpc.AgentStub(channel)

        # Get port of remote agent runner
        response = stub.SpawnWorker(agent_pb2.Machine())
        if response.status.result == "error":
            raise RemoteAgentException(
                "Agent runner could not be instantiated by remote zoo server."
            ) from e

        # Instantiate a local Remote Agent counterpart
        address = (ip, response.port)
        remote_agent = RemoteAgent(address)

        # Close channel
        channel.close()

        return remote_agent

    def _remote_agent_future(self):
        return self._replenish_threadpool.submit(
            self._build_remote_agent, self._zoo_master_addrs
        )

    def _try_to_acquire_remote_agent(self):
        assert len(self._agent_buffer) == self._buffer_size

        # Check if we have any done remote agent futures.
        done_future_indices = [
            idx
            for idx, agent_future in enumerate(self._agent_buffer)
            if agent_future.done()
        ]

        if len(done_future_indices) > 0:
            # If so, prefer one of these done ones to avoid sim delays.
            future = self._agent_buffer.pop(done_future_indices[0])
        else:
            # Otherwise, we will block, waiting on a remote agent future
            self._log.debug(
                "No ready remote agents, simulation will block until one is available"
            )
            future = self._agent_buffer.pop(0)

        # Schedule the next remote agent and add it to the buffer.
        self._agent_buffer.append(self._remote_agent_future())

        remote_agent = future.result(timeout=10)
        return remote_agent

    def acquire_remote_agent(self, retries=3) -> RemoteAgent:
        for i in range(retries):
            try:
                return self._try_to_acquire_remote_agent()
            except Exception as e:
                self._log.error(
                    f"Failed to acquire remote agent: {repr(e)} retrying {i} / {retries}"
                )
                time.sleep(0.1)

        raise Exception("Failed to acquire remote agent")
