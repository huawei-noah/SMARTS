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
import sys
import logging
import pathlib
import subprocess
import tempfile
import time
import atexit

from multiprocessing.connection import Client

import cloudpickle

from .agent import AgentSpec


class RemoteAgentException(Exception):
    pass


class RemoteAgent:
    def __init__(self, with_adaptation=True, connection_retries=100):
        atexit.register(self.terminate)

        self._log = logging.getLogger(self.__class__.__name__)
        sock_file = tempfile.mktemp()
        cmd = [
            sys.executable,  # path to the current python binary
            str(
                (pathlib.Path(__file__).parent.parent / "zoo" / "run_agent.py")
                .absolute()
                .resolve()
            ),
            sock_file,
        ]

        if with_adaptation:
            cmd.append("--with_adaptation")

        self._log.debug(f"Spawning remote agent proc: {cmd}")

        self._agent_proc = subprocess.Popen(cmd)
        self._conn = None

        for i in range(connection_retries):
            # Waiting on agent to open it's socket.
            try:
                self._conn = Client(sock_file, family="AF_UNIX")
                break
            except FileNotFoundError:
                self._log.debug(
                    f"RemoteAgent retrying connection to agent in: attempt {i}"
                )
                time.sleep(0.1)

        if self._conn is None:
            raise RemoteAgentException("Failed to connect to remote agent")

    def __del__(self):
        self.terminate()

    def send_observation(self, obs):
        self._conn.send({"type": "obs", "payload": obs})

    def recv_action(self, timeout=None):
        if self._conn.poll(timeout):
            try:
                return self._conn.recv()
            except ConnectionResetError as e:
                self.terminate()
                raise e
        else:
            return None

    def start(self, agent_spec: AgentSpec):
        # send the AgentSpec to the agent runner
        self._conn.send(
            # We use cloudpickle only for the agent_spec to allow for serialization of lambdas
            {"type": "agent_spec", "payload": cloudpickle.dumps(agent_spec)}
        )

    def terminate(self):
        if self._agent_proc:
            if self._conn:
                self._conn.close()

            if self._agent_proc.poll() is not None:
                self._agent_proc.kill()
                self._agent_proc.wait()
            self._agent_proc = None
