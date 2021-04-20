from smarts.core.agent import Agent, AgentSpec
from smarts.core.agent_interface import AgentInterface, AgentType


class KeepLaneAgent(Agent):
    def act(self, obs):
        return "keep_lane"


def create_agent_spec(max_episode_steps: int = None):

    return AgentSpec(
        interface=AgentInterface.from_type(
            AgentType.Laner,
            max_episode_steps=max_episode_steps,
        ),
        agent_builder=KeepLaneAgent,
    )
