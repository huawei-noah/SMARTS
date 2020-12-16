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
import cloudpickle
import logging
import time

from concurrent import futures
from multiprocessing.connection import Client

from .agent import AgentSpec


class RemoteAgentException(Exception):
    pass


class RemoteAgent:
    def __init__(self, address, socket_family, auth_key, connection_retries=100):
        self._log = logging.getLogger(self.__class__.__name__)
        auth_key_conn = str.encode(auth_key) if auth_key else None

        self._conn = None
        self._tp_exec = futures.ThreadPoolExecutor()

        for i in range(connection_retries):
            # Waiting on agent to open it's socket.
            try:
                self._conn = Client(
                    address, family=socket_family, authkey=auth_key_conn
                )
                break
            except Exception:
                self._log.debug(
                    f"RemoteAgent retrying connection to agent in: attempt {i}"
                )
                time.sleep(0.1)

        if self._conn is None:
            raise RemoteAgentException("Failed to connect to remote agent")

    def __del__(self):
        self.terminate()

    def _act(self, obs, timeout):
        # Send observation
        self._conn.send({"type": "obs", "payload": obs})
        # Receive action
        if self._conn.poll(timeout):
            try:
                return self._conn.recv()
            except ConnectionResetError as e:
                self.terminate()
                raise e
        else:
            return None

    def act(self, obs, timeout=None):
        # Run task asynchronously and return a Future
        return self._tp_exec.submit(self._act, obs, timeout)

    def start(self, agent_spec: AgentSpec):
        # Send the AgentSpec to the agent runner
        self._conn.send(
            # We use cloudpickle only for the agent_spec to allow for serialization of lambdas
            {"type": "agent_spec", "payload": cloudpickle.dumps(agent_spec)}
        )

    def terminate(self):
        if self._conn:
            self._conn.close()

        # Shutdown thread pool executor
        self._tp_exec.shutdown()
