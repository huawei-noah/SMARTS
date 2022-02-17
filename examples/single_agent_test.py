import time

import gym
import numpy as np

from examples.argument_parser import default_argument_parser
from smarts.core.agent import Agent
from smarts.core.utils.episodes import episodes
from smarts.env import build_scenario


class ChaseWaypointsAgent(Agent):
    def act(self, obs):
        cur_lane_index = obs.ego["lane_index"]
        next_lane_index = obs.waypoints["lane_index"][0, 0]

        return (
            obs.waypoints["speed_limit"][0, 0] / 4,
            np.sign(next_lane_index - cur_lane_index),
        )


def main(headless, num_episodes):
    env = gym.make(
        "smarts.env:intersection-v0",
        headless=headless,
        sumo_headless=True,
        visdom=False,
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

        print(
            "Score ==",
            info["score"],
            "Pos ==",
            observation.ego["pos"],
        )

    env.close()


if __name__ == "__main__":
    parser = default_argument_parser("single-agent-example")
    args = parser.parse_args()

    main(
        headless=True,  # args.headless,
        num_episodes=100,  # args.episodes,
    )
