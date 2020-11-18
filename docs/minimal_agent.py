import numpy as np

from smarts.zoo.registry import register
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.agent import Agent, AgentSpec
from smarts.core.controllers import ActionSpaceType


class BasicAgent(Agent):
    def act(self, obs):
        return "keep_lane"


register(
    locator="minimal",
    entry_point=lambda **kwargs: AgentSpec(
        interface=AgentInterface(waypoints=True, action=ActionSpaceType.Lane),
        agent_builder=BasicAgent,
    ),
)
