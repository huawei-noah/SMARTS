"""This is an example to show how SMARTS multi-agent works. This example uses the same kind of
agent multiple times. But different agents with different action and observation shapes can be mixed
in."""
import random
import sys
from pathlib import Path
from typing import Final

import gymnasium as gym

SMARTS_REPO_PATH = Path(__file__).parents[1].absolute()
sys.path.insert(0, str(SMARTS_REPO_PATH))
from examples.tools.argument_parser import minimal_argument_parser
from smarts.core.agent import Agent
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.utils.episodes import episodes
from smarts.sstudio.scenario_construction import build_scenarios

N_AGENTS = 4
AGENT_IDS: Final[list] = ["Agent %i" % i for i in range(N_AGENTS)]


class RandomLanerAgent(Agent):
    def __init__(self, action_space) -> None:
        self._action_space = action_space

    def act(self, obs, **kwargs):
        return self._action_space.sample()


class KeepLaneAgent(Agent):
    def __init__(self, action_space) -> None:
        self._action_space = action_space

    def act(self, obs, **kwargs):
        return self._action_space.sample()


def main(scenarios, headless, num_episodes, max_episode_steps=None):
    # This interface must match the action returned by the agent
    agent_interfaces = {
        agent_id: AgentInterface.from_type(
            AgentType.Laner, max_episode_steps=max_episode_steps
        )
        for agent_id in AGENT_IDS
    }

    env = gym.make(
        "smarts.env:hiway-v1",
        scenarios=scenarios,
        agent_interfaces=agent_interfaces,
        headless=headless,
    )

    for episode in episodes(n=num_episodes):
        agents = {
            agent_id: RandomLanerAgent(env.action_space[agent_id])
            for agent_id in agent_interfaces.keys()
        }
        observations, _ = env.reset()
        episode.record_scenario(env.unwrapped.scenario_log)

        terminateds = {"__all__": False}
        while not terminateds["__all__"]:
            actions = {
                agent_id: agent.act(observations) for agent_id, agent in agents.items()
            }
            observations, rewards, terminateds, truncateds, infos = env.step(actions)
            episode.record_step(observations, rewards, terminateds, truncateds, infos)

    env.close()


if __name__ == "__main__":
    parser = minimal_argument_parser(Path(__file__).stem)
    args = parser.parse_args()

    if not args.scenarios:
        args.scenarios = [
            str(SMARTS_REPO_PATH / "scenarios" / "sumo" / "loop"),
        ]

    build_scenarios(scenarios=args.scenarios)

    main(
        scenarios=args.scenarios,
        headless=args.headless,
        num_episodes=args.episodes,
        max_episode_steps=args.max_episode_steps,
    )
