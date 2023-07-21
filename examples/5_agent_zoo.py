"""This is an example to show how SMARTS multi-agent works. This example uses the same kind of
agent multiple times but different agents with different action and observation shapes can be mixed
in."""
import random
import sys
from pathlib import Path

from examples.tools.argument_parser import empty_parser

SMARTS_REPO_PATH = Path(__file__).parents[1].absolute()
sys.path.insert(0, str(SMARTS_REPO_PATH))
from smarts.core.agent import Agent
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.zoo import registry
from smarts.zoo.agent_spec import AgentSpec


class RandomLaneAgent(Agent):
    def act(self, obs, **kwargs):
        return random.randint(0, 3)


def rla_entrypoint(max_episode_steps=1000):
    return AgentSpec(
        interface=AgentInterface.from_type(
            AgentType.Laner, max_episode_steps=max_episode_steps
        ),
        agent_builder=RandomLaneAgent,
    )


def main():
    registry.register(
        "random_lane_control-v0", rla_entrypoint
    )  # This registers "__main__:random_lane_control-v0"
    print(registry.agent_registry)

    agent_spec = registry.make(locator="__main__:random_lane_control-v0")
    agent_interface = agent_spec.interface
    agent = agent_spec.build_agent()
    # alternatively this will build the agent
    agent, agent_interface = registry.make_agent(
        locator="__main__:random_lane_control-v0"
    )


if __name__ == "__main__":
    parser = empty_parser(Path(__file__).stem)
    args = parser.parse_args()

    main()
