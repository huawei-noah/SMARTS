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
import warnings
from typing import Any, Callable

from smarts.core.observations import Observation

warnings.simplefilter("once")

logger = logging.getLogger(__name__)


class Agent:
    """The base class for agents"""

    @classmethod
    def from_function(cls, agent_function: Callable[[Any], Any]) -> "Agent":
        """A utility function to create an agent from a lambda or other callable object.

        .. code-block:: python

            keep_lane_agent = Agent.from_function(lambda obs: "keep_lane")
        """
        assert callable(agent_function)

        class FunctionAgent(Agent):
            """An agent generated from a function."""

            def act(self, obs):
                return agent_function(obs)

        return FunctionAgent()

    def act(self, obs: Observation, **configs):
        """The agent action. See documentation on observations, `AgentSpec`, and `AgentInterface`.

        Expects an adapted observation and returns an unadapted action.
        """

        raise NotImplementedError


def deprecated_agent_spec(*args, **kwargs):
    """Deprecated version of AgentSpec, see smarts.zoo.agent_spec"""
    from smarts.zoo.agent_spec import AgentSpec as AgentSpecAlias

    warnings.warn(
        "The AgentSpec class has moved to the following module: smarts.zoo.agent_spec. Calling it from this module will be deprecated.",
        DeprecationWarning,
    )
    return AgentSpecAlias(*args, **kwargs)


AgentSpec = deprecated_agent_spec
