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
# FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import logging
import pathlib
import random
import signal
import subprocess
import sys
import time
from concurrent import futures
from typing import Optional, Tuple

import grpc

from smarts.core.agent_buffer import AgentBuffer
from smarts.core.buffer_agent import BufferAgent
from smarts.core.remote_agent import RemoteAgent, RemoteAgentException
from smarts.core.utils.networking import find_free_port
from smarts.zoo import manager_pb2, manager_pb2_grpc


class RemoteAgentBuffer(AgentBuffer):
    """A buffer that manages social agents."""

    def __init__(
        self,
        zoo_manager_addrs: Optional[Tuple[str, int]] = None,
        buffer_size: int = 3,
        max_workers: int = 4,
        timeout: float = 10,
    ):
        """Creates a local manager (if `zoo_manager_addrs=None`) or connects to a remote manager, to create and manage workers
        which execute agents.

        Args:
            zoo_manager_addrs (Optional[Tuple[str,int]], optional): List of (ip, port) tuples for manager processes. If none
                is provided, a manager is spawned in localhost. Defaults to None.
            buffer_size (int, optional): Number of RemoteAgents to pre-initialize and keep running in the background, must be
                non-zero. Defaults to 3.
            max_workers (int, optional): Maximum number of threads used for creation of workers. Defaults to 4.
            timeout (float, optional): Time (seconds) to wait for startup or response from server. Defaults to 10.
        """
        assert buffer_size > 0

        self._log = logging.getLogger(self.__class__.__name__)
        self._timeout = timeout

        # self._zoo_manager_conns is a list of dictionaries.
        # Each dictionary provides connection info for a zoo manager.
        # Example:
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
        self._zoo_manager_conns = []
        self._local_zoo_manager = False

        # Populate zoo manager connection with address and process handles.
        if not zoo_manager_addrs:
            # Spawn a local zoo manager since no remote zoo managers were provided.
            self._local_zoo_manager = True
            port = find_free_port()
            self._zoo_manager_conns = [
                {
                    "address": ("localhost", port),
                    "process": spawn_local_zoo_manager(port),
                }
            ]
        else:
            self._zoo_manager_conns = [{"address": addr} for addr in zoo_manager_addrs]

        # Populate zoo manager connection with channel and stub details.
        for conn in self._zoo_manager_conns:
            conn["channel"], conn["stub"] = get_manager_channel_stub(
                conn["address"], self._timeout
            )

        self._buffer_size = buffer_size
        self._replenish_threadpool = futures.ThreadPoolExecutor(max_workers=max_workers)
        self._agent_buffer = [
            self._remote_agent_future() for _ in range(self._buffer_size)
        ]

        # Catch abrupt terminate signals
        signal.signal(signal.SIGTERM, self._stop_servers)

    def _stop_servers(self, *args):
        self.destroy()
        self._log.debug(
            f"Shutting down zoo manager and zoo workers due to abrupt process stop."
        )
        sys.exit(0)

    def destroy(self):
        """Teardown any remaining remote agents and the local zoo manager (if it exists.)"""
        for remote_agent_future in self._agent_buffer:
            try:
                remote_agent = remote_agent_future.result()
                remote_agent.terminate()
            except Exception as e:
                self._log.error(
                    f"Exception while tearing down buffered remote agent. {repr(e)}"
                )
                raise e

        # If available, teardown local zoo manager.
        if self._local_zoo_manager:
            self._zoo_manager_conns[0]["channel"].close()
            self._zoo_manager_conns[0]["process"].terminate()
            self._zoo_manager_conns[0]["process"].wait()

    def _build_remote_agent(self, zoo_manager_conns):
        # Get a random zoo manager connection.
        zoo_manager_conn = random.choice(zoo_manager_conns)

        # Spawn remote worker and get its port.
        retries = 3
        worker_port = None
        for retry in range(retries):
            try:
                response = zoo_manager_conn["stub"].spawn_worker(manager_pb2.Machine())
                worker_port = response.num
                break
            except grpc.RpcError as e:
                self._log.debug(
                    f"Failed {retry+1}/{retries} times in attempt to spawn a remote worker process. {e}"
                )

        if worker_port == None:
            raise RemoteAgentException(
                "Remote worker process could not be spawned by the zoo manager."
            )

        # Instantiate and return a local RemoteAgent.
        return RemoteAgent(
            zoo_manager_conn["address"], (zoo_manager_conn["address"][0], worker_port)
        )

    def _remote_agent_future(self):
        return self._replenish_threadpool.submit(
            self._build_remote_agent, self._zoo_manager_conns
        )

    def _try_to_acquire_remote_agent(self, timeout: float):
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
                "No ready remote agents, simulation will block until one is available."
            )
            future = self._agent_buffer.pop(0)

        # Schedule the next remote agent and add it to the buffer.
        self._agent_buffer.append(self._remote_agent_future())

        remote_agent = future.result(timeout=timeout)
        return remote_agent

    def acquire_agent(
        self, retries: int = 3, timeout: Optional[float] = None
    ) -> BufferAgent:
        """Creates RemoteAgent objects.

        Args:
            retries (int, optional): Number of attempts in creating or connecting to an available
                RemoteAgent. Defaults to 3.
            timeout (Optional[float], optional): Time (seconds) to wait in acquiring a RemoteAgent.
                Defaults to None, which does not timeout.

        Raises:
            RemoteAgentException: If fail to acquire a RemoteAgent.

        Returns:
            RemoteAgent: A new RemoteAgent object.
        """
        if timeout == None:
            timeout = self._timeout

        for retry in range(retries):
            try:
                return self._try_to_acquire_remote_agent(timeout)
            except Exception as e:
                self._log.debug(
                    f"Failed {retry+1}/{retries} times in acquiring remote agent. {repr(e)}"
                )
                time.sleep(0.1)

        raise RemoteAgentException("Failed to acquire remote agent.")


def spawn_local_zoo_manager(port):
    """Generates a local manager subprocess."""
    cmd = [
        sys.executable,  # Path to the current Python binary.
        str(
            (pathlib.Path(__file__).parent.parent / "zoo" / "manager.py")
            .absolute()
            .resolve()
        ),
        "--port",
        str(port),
    ]

    manager = subprocess.Popen(cmd)
    if manager.poll() == None:
        return manager

    raise RuntimeError("Zoo manager subprocess is not running.")


def get_manager_channel_stub(addr: Tuple[str, int], timeout: float = 10):
    """Connects to the gRPC server at `addr` and returns the channel and stub.

    Args:
        addr (Tuple[str,int]): gRPC server address.
        timeout (float, optional): Time to wait for the gRPC server to be ready. Defaults to 10.

    Raises:
        RemoteAgentException: If timeout occurs while connecting to the gRPC server.

    Returns:
        grpc.Channel: Channel to the gRPC server.
        manager_pb2_grpc.ManagerStub : gRPC stub.
    """
    channel = grpc.insecure_channel(f"{addr[0]}:{addr[1]}")
    try:
        grpc.channel_ready_future(channel).result(timeout=timeout)
    except grpc.FutureTimeoutError:
        raise RemoteAgentException("Timeout in connecting to remote zoo manager.")
    stub = manager_pb2_grpc.ManagerStub(channel)
    return channel, stub
