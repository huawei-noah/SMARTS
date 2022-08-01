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
from concurrent import futures

from smarts.core.buffer_agent import BufferAgent
from smarts.zoo.agent_spec import AgentSpec


class LocalAgent(BufferAgent):
    """A local implementation of a buffer agent."""

    def __init__(self):
        self._agent = None
        self._agent_spec = None

    def act(self, obs):
        """Call the agent's act function asynchronously and return a Future."""

        def obtain_future_result(obs):
            adapted_obs = self._agent_spec.observation_adapter(obs)
            action = self._agent.act(adapted_obs)
            adapted_action = self._agent_spec.action_adapter(action)

            return adapted_action

        act_future = futures.Future()
        act_future.set_result(obtain_future_result(obs))
        return act_future

    def start(self, agent_spec: AgentSpec):
        """Send the AgentSpec to the agent runner."""
        self._agent_spec = agent_spec
        self._agent = self._agent_spec.build_agent()

    def terminate(self):
        pass
