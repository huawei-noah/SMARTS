from smarts.core.agent import Agent
from smarts.core.agent_interface import AgentInterface
from smarts.core.controllers import ActionSpaceType
from smarts.zoo.agent_spec import AgentSpec
from smarts.zoo.registry import register


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
