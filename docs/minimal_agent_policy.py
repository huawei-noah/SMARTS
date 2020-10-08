import numpy as np

from smarts.zoo.registry import register
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.agent import AgentPolicy, AgentSpec
from smarts.core.controllers import ActionSpaceType


class BasicPolicy(AgentPolicy):
    def act(self, obs):
        return "keep_lane"


register(
    locator="minimal",
    entry_point=lambda **kwargs: AgentSpec(
        interface=AgentInterface(waypoints=True, action=ActionSpaceType.Lane),
        policy_builder=BasicPolicy,
    ),
)
