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

from concurrent import futures
from multiprocessing import Process
from typing import List, Tuple

from smarts.core.remote_agent import RemoteAgent, RemoteAgentException
from smarts.core.utils.networking import find_free_port
from smarts.zoo import master as zoo_master
from smarts.zoo import master_pb2
from smarts.zoo import master_pb2_grpc


class RemoteAgentBuffer:
    def __init__(self, zoo_master_addrs=None, auth_key=None, buffer_size=3):
        """
        Args:
            zoo_master_addrs:
                List of (ip, port) tuples for master processes. Master processes instantiate 
                worker processes which run remote agents.
            auth_key:
                Authentication key of type string for communication.
            buffer_size: 
                Number of RemoteAgents to pre-initialize and keep running in the background, 
                must be non-zero (default: 3).
        """
        assert buffer_size > 0

        self._log = logging.getLogger(self.__class__.__name__)

        # List of dictionaries. Each dictionary provides connection info
        # for a zoo master. E.g.,
        # [
        #     {"address": ("127.0.0.1", 7432)),
        #      "process": <proc>,
        #      "channel": <grpc_channel>,
        #      "stub"   : <grpc_stub>
        #     },
        #     {
        #         ...
        #     }
        #     ...
        # ]
        self._zoo_master_conns = []
        self._local_zoo_master = False

        # Populate zoo master connection with address and process handles.
        if not zoo_master_addrs:
            # Spawn a local zoo master since no remote zoo masters was provided.
            self._local_zoo_master = True
            port = find_free_port()
            self._zoo_master_conns = [
                {
                    "address": ("localhost", port),
                    "process": spawn_local_zoo_master(port),
                }
            ]
        else:
            self._zoo_master_conns = [{"address": addr} for addr in zoo_master_addrs]

        # Populate zoo master connection with channel and stub details.
        for conn in self._zoo_master_conns:
            conn["channel"], conn["stub"] = get_master_channel_stub(conn["address"])

        self._buffer_size = buffer_size
        self._replenish_threadpool = futures.ThreadPoolExecutor()
        self._agent_buffer = [
            self._remote_agent_future() for _ in range(self._buffer_size)
        ]

        atexit.register(self.destroy)

    def destroy(self):
        if atexit:
            atexit.unregister(self.destroy)

        # Teardown any remaining remote agents.
        for remote_agent_future in self._agent_buffer:
            try:
                remote_agent = remote_agent_future.result()
                remote_agent.terminate()
            except Exception as e:
                self._log.error(
                    f"Exception while tearing down buffered remote agent: {repr(e)}"
                )

        # If available, teardown local zoo master.
        if self._local_zoo_master:
            self._zoo_master_conns[0]["channel"].close()
            self._zoo_master_conns[0]["process"].terminate()
            self._zoo_master_conns[0]["process"].join()

    def _build_remote_agent(self, zoo_master_conns):
        # Get a random zoo master connection.
        zoo_master_conn = random.choice(zoo_master_conns)

        # Spawn remote worker and get its port.
        retries = 5
        worker_port = None
        for ii in range(retries):
            response = zoo_master_conn["stub"].SpawnWorker(master_pb2.Machine())
            if response.status.code == 0:
                worker_port = response.port
                break
            self._log.error(
                f"Failed {ii+1}/{retries} times in attempt to spawn a remote worker process."
            )
        if worker_port == None:
            raise RemoteAgentException(
                "Remote worker process could not be instantiated by master process."
            )

        # Instantiate and return a local RemoteAgent.
        return RemoteAgent(zoo_master_conn["address"], ("localhost", worker_port))

    def _remote_agent_future(self):
        return self._replenish_threadpool.submit(
            self._build_remote_agent, self._zoo_master_conns
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
            # Otherwise, we will block, waiting on a remote agent future.
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


def spawn_local_zoo_master(port):
    master = Process(target=zoo_master.serve, args=(port,))
    master.start()
    return master


def get_master_channel_stub(addr):
    channel = grpc.insecure_channel(f"{addr[0]}:{addr[1]}")
    try:
        # Wait until the grpc server is ready or timeout after 30 seconds
        grpc.channel_ready_future(channel).result(timeout=30)
    except grpc.FutureTimeoutError:
        raise RemoteAgentException(
            "Timeout in connecting to remote zoo master server."
        ) from e
    stub = master_pb2_grpc.MasterStub(channel)
    return channel, stub
