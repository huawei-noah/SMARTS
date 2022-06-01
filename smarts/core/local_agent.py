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
import time
from concurrent import futures
from typing import Tuple
from buffer_agent import BufferAgent
from agent import Agent

import cloudpickle

from smarts.zoo import manager_pb2, worker_pb2
from smarts.zoo.agent_spec import AgentSpec


class LocalAgent(BufferAgent):
    """A remotely controlled agent."""

    def __init__(
        self
    ):
        """Executes an agent in a worker (i.e., a gRPC server).

        Args:
            manager_address (Tuple[str,int]): Manager's server address (ip, port).
            worker_address (Tuple[str,int]): Worker's server address (ip, port).
            timeout (float, optional): Time (seconds) to wait for startup or response from
                server. Defaults to 10.

        Raises:
            RemoteAgentException: If timeout occurs while connecting to the manager or worker.
        """

        # Track the last action future.
        self._act_future = None

    def act(self, obs):
        """Call the agent's act function asynchronously and return a Future."""
        self._act_future = concurrent.future.Future()      
        _act_future.set_result(worker_pb2.Observation(payload=cloudpickle.dumps(obs)))

        return self._act_future

    def start(self, agent_spec: AgentSpec):
        """Send the AgentSpec to the agent runner."""
        # Cloudpickle used only for the agent_spec to allow for serialization of lambdas.
        worker_pb2.Specification(payload=cloudpickle.dumps(agent_spec))
        

    def terminate(self):
        pass
