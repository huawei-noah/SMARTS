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

from multiprocessing import Process
from multiprocessing.connection import Client
from concurrent import futures

from smarts.zoo import worker as zoo_worker
from .remote_agent import RemoteAgent, RemoteAgentException
from .utils.networking import find_free_port


class RemoteAgentBuffer:
    def __init__(self, zoo_worker_addrs=None, auth_key=None, buffer_size=3):
        """
        Args:
            zoo_worker_addrs:
                List of (ip, port) tuples of Zoo Workers, used to instantiate remote social agents
            auth_key:
                Authentication key of type string for communication with Zoo Workers
            buffer_size: 
                Number of RemoteAgents to pre-initialize and keep running in the background, must be non-zero (default: 3)
        """
        assert buffer_size > 0

        self._log = logging.getLogger(self.__class__.__name__)

        self._local_zoo_worker = None
        self._local_zoo_worker_addr = None

        assert isinstance(
            auth_key, (str, type(None))
        ), f"Received auth_key of type {type(auth_key)}, but need auth_key of type <class 'string'> or <class 'NoneType'>."
        self._auth_key = auth_key if auth_key else ""
        self._auth_key_conn = str.encode(auth_key) if auth_key else None

        if zoo_worker_addrs is None:
            # The user has not specified any remote zoo workers.
            # We need to spawn a local zoo worker.
            self._local_zoo_worker_addr = self._spawn_local_zoo_worker()
            zoo_worker_addrs = [self._local_zoo_worker_addr]

        self._zoo_worker_addrs = zoo_worker_addrs

        self._buffer_size = buffer_size
        self._replenish_threadpool = futures.ThreadPoolExecutor()
        self._agent_buffer = [self._remote_agent_future() for _ in range(buffer_size)]

        atexit.register(self.destroy)

    def destroy(self):
        if atexit:
            atexit.unregister(self.destroy)

        for remote_agent_future in self._agent_buffer:
            # try to cancel the future
            if not remote_agent_future.cancel():
                # We can't cancel this future, wait for it to complete
                # and terminate the agent after it's been created
                try:
                    remote_agent = remote_agent_future.result()
                    remote_agent.terminate()
                except Exception as e:
                    self._log.error(
                        f"Exception while tearing down buffered remote agent: {repr(e)}"
                    )

        # Note: important to teardown the local zoo worker after we purge the remote agents
        #       since they may still require the local zoo worker to for instantiation.
        if self._local_zoo_worker is not None:
            self._local_zoo_worker.kill()
            self._local_zoo_worker.join()

    def _spawn_local_zoo_worker(self, retries=3):
        local_port = find_free_port()
        self._local_zoo_worker = Process(
            target=zoo_worker.listen, args=(local_port, self._auth_key)
        )
        self._local_zoo_worker.start()

        local_address = ("0.0.0.0", local_port)

        # Block until the local zoo server is accepting connections.
        for i in range(retries):
            try:
                conn = Client(
                    local_address, family="AF_INET", authkey=self._auth_key_conn
                )
                break
            except Exception as e:
                self._log.error(
                    f"Waiting for local zoo worker to start up, retrying {i} / {retries}"
                )
                time.sleep(0.1)

        return local_address

    def _try_to_allocate_remote_agent(self, zoo_worker_addr, conn):
        address, family, resp = None, None, None
        if zoo_worker_addr == self._local_zoo_worker_addr:
            # This is a local zoo worker, as an optimization, allocate a local remote agent
            conn.send("allocate_local_agent")
            resp = conn.recv()
            if resp["result"] == "success":
                address = resp["socket_file"]
                family = "AF_UNIX"
        else:
            # This is a remote zoo worker, we need to communicate with this
            # remote agent over a network socket
            conn.send("allocate_networked_agent")
            resp = conn.recv()
            if resp["result"] == "success":
                port = resp["port"]
                address = (zoo_worker_addr[0], port)
                family = "AF_INET"

        if address is None or family is None:
            self._log.error(
                f"Failed to allocate remote agent on {zoo_worker_addr} {repr(resp)}"
            )
            return None

        self._log.debug(f"Connecting to remote agent at {address} with {family}")
        return RemoteAgent(address, family, self._auth_key)

    def _remote_agent_future(self, retries=5):
        def build_remote_agent():
            for i in range(retries):
                conn = None
                try:
                    # Try to connect to a random zoo worker.
                    zoo_worker_addr = random.choice(self._zoo_worker_addrs)
                    conn = Client(
                        zoo_worker_addr, family="AF_INET", authkey=self._auth_key_conn
                    )

                    # Now that we've got a connection to this zoo worker,
                    # request an allocation.
                    remote_agent = self._try_to_allocate_remote_agent(
                        zoo_worker_addr, conn
                    )

                    if remote_agent is not None:
                        # We were successful in acquiring a remote agent, return it.
                        return remote_agent
                except Exception as e:
                    self._log.error(
                        f"Failed to connect to {zoo_worker_addr}, retrying {i} / {retries} '{e} {repr(e)}'"
                    )
                finally:
                    if conn:
                        conn.close()

                # Otherwise sleep and retry.
                time.sleep(0.5)
                self._log.error(
                    f"Failed to allocate agent on {zoo_worker_addr}, retrying {i} / {retries}"
                )

            raise Exception("Failed to allocate remote agent")

        return self._replenish_threadpool.submit(build_remote_agent)

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
