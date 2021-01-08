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
from typing import Optional, Any, Callable
from dataclasses import dataclass, replace
import warnings

import cloudpickle

from .agent_interface import AgentInterface

warnings.simplefilter("once")

logger = logging.getLogger(__name__)


class Agent:
    """The base class for agents"""

    @classmethod
    def from_function(cls, agent_function: Callable):
        """A utility function to create an agent from a lambda or other callable object.

        .. code-block:: python

            keep_lane_agent = Agent.from_function(lambda obs: "keep_lane")
        """
        assert callable(agent_function)

        class FunctionAgent(Agent):
            def act(self, obs):
                return agent_function(obs)

        return FunctionAgent()

    def act(self, obs):
        """The agent action. See documentation on observations, `AgentSpec`, and `AgentInterface`.

        Expects an adapted observation and returns an unadapted action.
        """

        raise NotImplementedError


# Remain backwards compatible with existing Agent's
class AgentPolicy(Agent):
    # we cannot use debtcollector here to signal deprecation because the wrapper object is
    # not pickleable, instead we simply print.
    def __init__(self):
        logger.warning(
            "[DEPRECATED] AgentPolicy has been replaced with `smarts.core.agent.Agent`"
        )


@dataclass
class AgentSpec:
    """A configuration that is used by SMARTS environments.

    .. code-block:: python

        agent_spec = AgentSpec(
            interface=AgentInterface.from_type(AgentType.Laner),
            agent_params={"agent_function": lambda _: "keep_lane"},
            agent_builder=AgentPolicy.from_function,
        )

        env = gym.make(
            "smarts.env:hiway-v0",
            scenarios=["scenarios/loop"],
            agent_specs={agent_id: agent_spec},
        )

        agent = agent_spec.build_agent()

    Refer to the Agent documentation.
    """

    # This is optional because sometimes when building re-useable specs,
    # you don't know the agent interface ahead of time.
    interface: AgentInterface = None
    """the adaptor used to wrap agent observation and action flow (default None)"""

    agent_builder: Callable[..., Agent] = None
    """A callable to build an `smarts.core.agent.Agent` given `AgentSpec.agent_params` (default None)"""
    agent_params: Optional[Any] = None
    """Parameters to be given to `AgentSpec.agent_builder` (default None)"""

    policy_builder: Callable[..., Agent] = None
    """[DEPRECATED] see `AgentSpec.agent_builder` (default None)"""
    policy_params: Optional[Any] = None
    """[DEPRECATED] see `AgentSpec.agent_params` (default None)"""
    observation_adapter: Callable = lambda obs: obs
    """An adaptor that allows shaping of the observations (default lambda obs: obs)"""
    action_adapter: Callable = lambda act: act
    """An adaptor that allows shaping of the action (default lambda act: act)"""
    reward_adapter: Callable = lambda obs, reward: reward
    """An adaptor that allows shaping of the reward (default lambda obs, reward: reward)"""
    info_adapter: Callable = lambda obs, reward, info: info
    """An adaptor that allows shaping of info (default lambda obs, reward, info: info)"""
    perform_self_test: bool = False
    """[DEPRECATED] this parameter is not used anymore"""

    def __post_init__(self):
        # make sure we can pickle ourselves
        cloudpickle.dumps(self)

        if self.perform_self_test:
            logger.warning(
                f"[DEPRECATED] `perform_self_test` has been deprecated: {self}"
            )

        if self.policy_builder:
            logger.warning(
                f"[DEPRECATED] Please use AgentSpec(agent_builder=<...>) instead of AgentSpec(policy_builder=<..>):\n {self}"
                "policy_builder will overwrite agent_builder"
            )
            self.agent_builder = self.policy_builder

        if self.policy_params:
            logger.warning(
                f"[DEPRECATED] Please use AgentSpec(agent_params=<...>) instead of AgentSpec(policy_params=<..>):\n {self}"
                "policy_builder will overwrite agent_builder"
            )
            self.agent_params = self.policy_params

        self.policy_params = self.agent_params
        self.policy_builder = self.agent_builder

    def replace(self, **kwargs) -> "AgentSpec":
        """Return a copy of this AgentSpec with the given fields updated."""

        replacements = [
            ("policy_builder", "agent_builder"),
            ("policy_params", "agent_params"),
            ("perform_self_test", None),
        ]

        assert (
            None not in kwargs
        ), f"Error: kwargs input to replace() function contains invalid key `None`: {kwargs}"

        kwargs_copy = kwargs.copy()
        for deprecated, current in replacements:
            if deprecated in kwargs:
                if current:
                    logger.warning(
                        f"[DEPRECATED] Please use AgentSpec.replace({current}=<...>) instead of AgentSpec.replace({deprecated}=<...>)\n"
                    )
                else:
                    logger.warning(
                        f"[DEPRECATED] Attribute {deprecated} no longer has effect."
                    )
                assert (
                    current not in kwargs
                ), f"Mixed current ({current}) and deprecated ({deprecated}) values in replace"
                moved = kwargs[deprecated]
                del kwargs_copy[deprecated]
                if current:
                    kwargs_copy[current] = moved

        return replace(self, **kwargs_copy)

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
            return self.agent_builder(**self.agent_params)
        else:
            # otherwise, the agent params are sent as is to the builder
            return self.agent_builder(self.agent_params)
