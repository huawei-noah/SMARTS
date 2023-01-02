# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
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
import inspect
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Optional

import cloudpickle

from smarts.core.agent import Agent
from smarts.core.agent_interface import AgentInterface

warnings.simplefilter("once")


@dataclass
class AgentSpec(object):
    """A configuration that is used by SMARTS environments.

    .. code-block:: python

        agent_spec = AgentSpec(
            interface=AgentInterface.from_type(AgentType.Laner),
            agent_params={"agent_function": lambda _: "keep_lane"},
            agent_builder=Agent.from_function,
        )

        env = gym.make(
            "smarts.env:hiway-v0",
            scenarios=["scenarios/sumo/loop"],
            agent_specs={agent_id: agent_spec},
        )

        agent = agent_spec.build_agent()

    Refer to the Agent documentation.
    """

    # This is optional because sometimes when building re-useable specs,
    # you don't know the agent interface ahead of time.
    interface: Optional[AgentInterface] = None
    """the adaptor used to wrap agent observation and action flow (default None)"""

    agent_builder: Optional[Callable[..., Agent]] = None
    """A callable to build an `smarts.core.agent.Agent` given `AgentSpec.agent_params` (default None)"""
    agent_params: Optional[Any] = None
    """Parameters to be given to `AgentSpec.agent_builder` (default None)"""

    observation_adapter: Callable = lambda obs: obs
    """Deprecated. Do not use. An adaptor that allows shaping of the 
    observations. Defaults to `lambda obs: obs`."""
    action_adapter: Callable = lambda act: act
    """Deprecated. Do not use. An adaptor that allows shaping of the action. 
    Defaults to `lambda act: act`."""
    reward_adapter: Callable = lambda obs, reward: reward
    """Deprecated. Do not use. An adaptor that allows shaping of the reward. 
    Defaults to `lambda obs, reward: reward`."""
    info_adapter: Callable = lambda obs, reward, info: info
    """Deprecated. Do not use. An adaptor that allows shaping of info. Defaults
    to `lambda obs, reward, info: info`."""

    def __getattribute__(self, item):
        if (
            item == "observation_adapter"
            or item == "action_adapter"
            or item == "reward_adapter"
            or item == "info_adapter"
        ):
            warnings.warn(
                "Observation_adapter, action_adapter, reward_adapter, "
                "and info_adapter, are deprecated. Do not use them.",
                DeprecationWarning,
                stacklevel=2,
            )
        return object.__getattribute__(self, item)

    def __post_init__(self):
        # make sure we can pickle ourselves
        cloudpickle.dumps(self)

    def build_agent(self) -> Agent:
        """Construct an Agent from the AgentSpec configuration."""
        if self.agent_builder is None:
            raise ValueError("Can't build agent, no agent builder was supplied")

        if not callable(self.agent_builder):
            raise ValueError(
                f"""agent_builder: {self.agent_builder} is not callable
Use a combination of agent_params and agent_builder to define how to build your agent, ie.

AgentSpec(
  agent_params={{"input_dimensions": 12}},
  agent_builder=MyAgent # we are not instantiating the agent, just passing the class reference
)
"""
            )

        if self.agent_params is None:
            # no args to agent builder
            return self.agent_builder()
        elif isinstance(self.agent_params, (list, tuple)):
            # a list or tuple is treated as positional arguments
            return self.agent_builder(*self.agent_params)
        elif isinstance(self.agent_params, dict):
            # dictionaries, as keyword arguments
            fas = inspect.getfullargspec(self.agent_builder)
            if fas[2] is not None:
                return self.agent_builder(**self.agent_params)
            else:
                return self.agent_builder(
                    **{
                        k: self.agent_params[k]
                        for k in self.agent_params.keys() & set(fas[0])
                    }
                )
        else:
            # otherwise, the agent params are sent as is to the builder
            return self.agent_builder(self.agent_params)
