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
from typing import Optional, Any, Callable
from dataclasses import dataclass, replace

import cloudpickle

from .agent_interface import AgentInterface


class AgentPolicy:
    """The base class for agent policies."""

    @classmethod
    def from_function(cls, policy_function: Callable):
        """A utility function to create a policy from a lambda or other callable object.

        .. code-block:: python

            keep_lane_policy = AgentPolicy.from_function(lambda obs: "keep_lane")
        """
        assert callable(policy_function)

        class FunctionPolicy(AgentPolicy):
            def act(self, obs):
                return policy_function(obs)

        return FunctionPolicy()

    def act(self, obs):
        """The policy action. See documentation on observations, `Agent`, and `AgentInterface`.

        Expects an adapted observation and returns an unadapted action.
        """

        raise NotImplementedError


@dataclass
class Agent:
    """The core agent class. This gathers the interface, policy, and adapters."""

    interface: AgentInterface
    """The adaptor used to wrap agent observation and action flow"""
    policy: AgentPolicy
    """The wrapper for the policy action"""
    observation_adapter: Callable
    """An adaptor that allows shaping of the observations"""
    reward_adapter: Callable
    """An adaptor that allows shaping of the reward"""
    action_adapter: Callable
    """An adaptor that allows shaping of the action"""
    info_adapter: Callable
    """An adaptor that allows shaping of info"""

    def act(self, observation):
        """Calls the policy action. Expects adapted observation and returns an unadapted action.

        See documentation on observations and `AgentInterface`.
        """

        assert self.policy, "Unable to call Agent.act(...) without a policy"
        action = self.policy.act(observation)
        return action

    # XXX: This method should only be used by social agents, not by ego agents.
    def act_with_adaptation(self, env_observation):
        observation = self.observation_adapter(env_observation)
        policy_action = self.act(observation)
        action = self.action_adapter(policy_action)
        return action


@dataclass
class AgentSpec:
    """A configuration that is used by SMARTS environments.

    .. code-block:: python

        agent_spec = AgentSpec(
            interface=AgentInterface.from_type(AgentType.Laner),
            policy_params={"policy_function": lambda _: "keep_lane"},
            policy_builder=AgentPolicy.from_function,
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

    # If you are training a policy with RLLib, you don't necessarily
    # want to set the policy as part of the AgentSpec, thus we leave
    # it as an optional field.
    policy_builder: Callable[..., AgentPolicy] = None
    """An adaptor that interprets the policy_params into a policy  (default None)"""
    policy_params: Optional[Any] = None
    """Parameters to be fed into the policy builder (default None)"""
    observation_adapter: Callable = lambda obs: obs
    """An adaptor that allows shaping of the observations (default lambda obs: obs)"""
    action_adapter: Callable = lambda act: act
    """An adaptor that allows shaping of the action (default lambda act: act)"""
    reward_adapter: Callable = lambda obs, reward: reward
    """An adaptor that allows shaping of the reward (default lambda obs, reward: reward)"""
    info_adapter: Callable = lambda obs, reward, info: info
    """An adaptor that allows shaping of info (default lambda obs, reward, info: info)"""
    perform_self_test: bool = True

    def __post_init__(self):
        # make sure we can pickle ourselves
        cloudpickle.dumps(self)
        # Perform a self-test
        # TODO: move this to a remote agent, this is not safe to do in the smarts process
        if self.policy_builder is None:
            # skip this self-test if we have not set a `policy_builder`
            return

        if self.perform_self_test:
            policy = self._build_policy()
            assert isinstance(
                policy, AgentPolicy
            ), f"Policy builder did not build an AgentPolicy: {policy}"
            # TODO: The user has to hook up a few things correctly here
            #       and without types to guide the user, we should help
            #       them proactively.
            #
            #       perform a self-test here to ensure things are rigged
            #       up properly. ie.  generate some fake obs based on the
            #       interface, pass it through the obs adapter, run that
            #       through the policy, pass the policy action through
            #       the action_adapter and ensure the controller accepts
            #       the produced action.
            del policy

    def replace(self, **kwargs) -> "AgentSpec":
        """Return a copy of this AgentSpec with the given fields updated."""

        return replace(self, **kwargs)

    def build_agent(self) -> Agent:
        """Construct an Agent from the AgentSpec configuration."""

        return Agent(
            interface=self.interface,
            observation_adapter=self.observation_adapter,
            action_adapter=self.action_adapter,
            reward_adapter=self.reward_adapter,
            info_adapter=self.info_adapter,
            policy=self._build_policy(),
        )

    def _build_policy(self):
        if self.policy_builder is None:
            raise ValueError("Can't build agent, no policy builder was supplied")

        if not callable(self.policy_builder):
            raise ValueError(
                f"""policy_builder: {self.policy_builder} is not callable
Use a combination of policy_params and policy_builder to define how to build your policy, ie.
AgentSpec(
  policy_params={{"input_dimensions": 12}},
  policy_builder=lambda: input_dimensions: MyPolicy(input_dimensions)
)

OR Better yet

AgentSpec(
  policy_params={{"input_dimensions": 12}},
  policy_builder=MyPolicy # we are not instantiating the policy, just passing the class reference
)
"""
            )

        if self.policy_params is None:
            # no args to policy builder
            return self.policy_builder()
        elif isinstance(self.policy_params, (list, tuple)):
            # a list or tuple is treated as positional arguments
            return self.policy_builder(*self.policy_params)
        elif isinstance(self.policy_params, dict):
            # dictionaries, as keyword arguments
            return self.policy_builder(**self.policy_params)
        else:
            # otherwise, the policy params are sent as is to the builder
            return self.policy_builder(self.policy_params)
