import random
import sys
from pathlib import Path

import gym

sys.path.insert(0, str(Path(__file__).parents[2].absolute()))
from examples.tools.argument_parser import default_argument_parser
from smarts.core.agent import Agent
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.utils.episodes import episodes
from smarts.sstudio.scenario_construction import build_scenarios
from smarts.zoo.agent_spec import AgentSpec

N_AGENTS = 4
AGENT_IDS = ["Agent %i" % i for i in range(N_AGENTS)]


class KeepLaneAgent(Agent):
    def act(self, obs):
        val = ["keep_lane", "slow_down", "change_lane_left", "change_lane_right"]
        return random.choice(val)


def main(scenarios, headless, num_episodes, max_episode_steps=None):
    agent_specs = {
        agent_id: AgentSpec(
            interface=AgentInterface.from_type(
                AgentType.Laner, max_episode_steps=max_episode_steps
            ),
            agent_builder=KeepLaneAgent,
        )
        for agent_id in AGENT_IDS
    }

    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=scenarios,
        agent_interfaces={
            a_id: a_intrf.interface for a_id, a_intrf in agent_specs.items()
        },
        headless=headless,
        sumo_headless=True,
    )

    for episode in episodes(n=num_episodes):
        agents = {
            agent_id: agent_spec.build_agent()
            for agent_id, agent_spec in agent_specs.items()
        }
        observations = env.reset()
        episode.record_scenario(env.scenario_log)

        dones = {"__all__": False}
        while not dones["__all__"]:
            actions = {
                agent_id: agents[agent_id].act(agent_obs)
                for agent_id, agent_obs in observations.items()
            }
            observations, rewards, dones, infos = env.step(actions)
            episode.record_step(observations, rewards, dones, infos)

    env.close()


if __name__ == "__main__":
    parser = default_argument_parser("laner")
    args = parser.parse_args()

    if not args.scenarios:
        args.scenarios = [
            str(Path(__file__).absolute().parents[2] / "scenarios" / "sumo" / "loop"),
            str(
                Path(__file__).absolute().parents[2]
                / "scenarios"
                / "sumo"
                / "figure_eight"
            ),
        ]

    build_scenarios(scenarios=args.scenarios)

    main(
        scenarios=args.scenarios,
        headless=args.headless,
        num_episodes=args.episodes,
        max_episode_steps=args.max_episode_steps,
    )
