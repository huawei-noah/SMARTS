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
from concurrent import futures

from threading import Lock, Thread

from .remote_agent import RemoteAgent, RemoteAgentException


class RemoteAgentBuffer:
    def __init__(self, zoo_worker_addrs=None, buffer_size=3):
        """
        Args:
          buffer_size: Number of RemoteAgents to pre-initialize and keep running in the background, must be non-zero (default: 3)
        """
        assert buffer_size > 0

        self._log = logging.getLogger(self.__class__.__name__)

        self._local_zoo_worker = None
        self._local_worker_addr = None
        if zoo_worker_addrs is None:
            self._local_worker_addr = self._spawn_local_zoo_worker()
            zoo_worker_addrs = [self._local_worker_addr]
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

        if self._local_zoo_worker is not None:
            self._local_zoo_worker.kill()
            self._local_zoo_worker.join()

    def _spawn_local_zoo_worker(self):
        from multiprocessing import Process
        from smarts.zoo.worker import listen
        from smarts.core.utils.networking import find_free_port

        port = find_free_port()
        self._local_zoo_worker = Process(target=listen, args=(port,))
        self._local_zoo_worker.start()
        return ("0.0.0.0", port)

    def _remote_agent_future(self, retries=32):
        def build_remote_agent():
            import random
            import time

            from multiprocessing.connection import Client

            for i in range(retries):
                conn = None
                try:
                    zoo_worker_addr = random.choice(self._zoo_worker_addrs)
                    conn = Client(zoo_worker_addr, family="AF_INET")

                    if zoo_worker_addr == self._local_worker_addr:
                        conn.send("allocate_local_agent")
                        resp = conn.recv()
                        if resp["result"] == "success":
                            address = resp["socket_file"]
                            family = "AF_UNIX"
                        else:
                            time.sleep(0.5)
                            self._log.error(
                                f"Failed to allocate agent on {zoo_worker_addr}: {resp}, retrying {i} / {retries}"
                            )
                            continue
                    else:
                        conn.send("allocate_networked_agent")
                        resp = conn.recv()
                        if resp["result"] == "success":
                            port = resp["port"]
                            address = (zoo_worker_addr[0], port)
                            family = "AF_INET"
                        else:
                            time.sleep(0.5)
                            self._log.error(
                                f"Failed to allocate agent on {zoo_worker_addr}: {resp}, retrying {i} / {retries}"
                            )
                            continue
                    assert address is not None
                    assert family is not None
                    self._log.info(f"Connecting to remote agent at {address} {family}")
                    return RemoteAgent(address, family)
                    break
                except Exception as e:
                    self._log.error(
                        f"Failed to connect to {zoo_worker_addr}, retrying {i} / {retries} '{e} {repr(e)}'"
                    )
                finally:
                    if conn:
                        conn.close()
            raise Exception("Failed to allocate remote agent")

        return self._replenish_threadpool.submit(build_remote_agent)

    def acquire_remote_agent(self, retries=3) -> RemoteAgent:
        for i in range(retries):
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

            try:
                remote_agent = future.result(timeout=10)
                return remote_agent
            except Exception as e:
                self._log.error(
                    f"Failed to acquire remote agent: {e} retrying {i} / {retries}"
                )
                time.sleep(0.1)

        raise Exception("Failed to acquire remote agent")
