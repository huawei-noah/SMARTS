import numpy as np

from smarts.core.agent import Agent
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.controllers import ActionSpaceType
from smarts.zoo.agent_spec import AgentSpec
from smarts.zoo.registry import register


class DummyAgent(Agent):
    def act(self, obs):
        linear_acceleration = 0
        angular_vel = 0
        return [0, 0]


register(
    locator="interface-dummy-imitation-v0",
    entry_point=lambda **kwargs: AgentSpec(
        interface=AgentInterface.from_type(AgentType.Direct),
        agent_builder=DummyAgent,
    ),
)
