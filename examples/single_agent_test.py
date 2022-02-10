import logging
import pathlib

import gym
import numpy as np

from examples.argument_parser import default_argument_parser
from smarts.core.agent import Agent, AgentSpec
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.utils.episodes import episodes
from smarts.env import build_scenario
from smarts.env.wrappers.single_agent import SingleAgent

logging.basicConfig(level=logging.INFO)


class ChaseWaypointsAgent(Agent):
    def act(self, obs):
        cur_lane_index = obs.ego["lane_index"]
        next_lane_index = obs.waypoints["lane_index"][0, 0]

        return (
            obs.waypoints["speed_limit"][0, 0] / 2,
            np.sign(next_lane_index - cur_lane_index),
        )


def main(headless, num_episodes):
    env = gym.make(
        "smarts.env:intersection-v0",
        headless=headless,
        sumo_headless=True,
    )

    for episode in episodes(n=num_episodes):
        agent = ChaseWaypointsAgent()
        observation = env.reset()
        episode.record_scenario(env.scenario_log)

        done = False
        while not done:
            agent_action = agent.act(observation)
            observation, reward, done, info = env.step(agent_action)
            episode.record_step(observation, reward, done, info)

    env.close()


if __name__ == "__main__":
    parser = default_argument_parser("single-agent-example")
    args = parser.parse_args()

    if not args.scenarios:
        args.scenarios = [
            str(
                pathlib.Path(__file__).absolute().parents[1]
                / "scenarios"
                / "loop"
            )
        ]

    build_scenario(args.scenarios)

    main(
        headless=args.headless,
        num_episodes=args.episodes,
    )
