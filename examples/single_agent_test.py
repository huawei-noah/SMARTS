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
            obs.waypoints["speed_limit"][0, 0] / 5,
            np.sign(next_lane_index - cur_lane_index),
        )


def main(headless, num_episodes):
    env = gym.make(
        "smarts.env:intersection-v0",
        headless=True,
        sumo_headless=True,
        visdom=True,
    )

    for episode in episodes(n=num_episodes):
        agent = ChaseWaypointsAgent()
        observation = env.reset()
        episode.record_scenario(env.scenario_log)

        tot_reward = 0
        done = False
        while not done:
            agent_action = agent.act(observation)
            observation, reward, done, info = env.step(agent_action)
            episode.record_step(observation, reward, done, info)
            tot_reward += reward
            # if tot_reward >= 100 and tot_reward<=101:
            #     print("TOTAL REWARD EXCEEDED 100")
            #     time.sleep(10)
        print("Score==", info["score"], ", reward==", tot_reward)

    env.close()


if __name__ == "__main__":
    parser = default_argument_parser("single-agent-example")
    args = parser.parse_args()

    main(
        headless=args.headless,
        num_episodes=args.episodes,
    )
