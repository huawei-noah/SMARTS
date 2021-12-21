from smarts.core.agent import Agent, AgentSpec
from smarts.core.agent_interface import AgentInterface, DoneCriteria
from smarts.core.controllers import ActionSpaceType
from smarts.zoo.registry import register


class KeepLaneAgent(Agent):
    def act(self, obs):
        return "keep_lane"


class StoppedAgent(Agent):
    def act(self, obs):
        return (0, 0)


register(
    locator="laner-agent-v0",
    entry_point=lambda **kwargs: AgentSpec(
        interface=AgentInterface(
            waypoints=True, action=ActionSpaceType.Lane, max_episode_steps=5000
        ),
        agent_builder=KeepLaneAgent,
    ),
)

register(
    locator="buddha-agent-v0",
    entry_point=lambda **kwargs: AgentSpec(
        interface=AgentInterface(
            waypoints=True,
            action=ActionSpaceType.LaneWithContinuousSpeed,
            done_criteria=DoneCriteria(not_moving=True),
        ),
        agent_builder=StoppedAgent,
    ),
)
