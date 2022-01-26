import logging
import pathlib
import subprocess
from typing import List

import gym
import numpy as np

from smarts.core.agent import Agent, AgentSpec
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.utils.episodes import episodes
from smarts.env.wrappers.single_agent import SingleAgent
from smarts.env.wrappers.standard_obs import StandardObs, StdObs

if __name__ == "__main__":
    from argument_parser import default_argument_parser
else:
    from .argument_parser import default_argument_parser

logging.basicConfig(level=logging.INFO)


class ChaseWaypointsAgent(Agent):
    def act(self, obs: StdObs):
        cur_lane_index = obs.ego["lane_index"]
        next_lane_index = obs.waypoints["lane_index"][0, 0]

        return (
            obs.waypoints["speed_limit"][0, 0] / 2,
            np.sign(next_lane_index - cur_lane_index),
        )


def main(scenarios, headless, num_episodes, max_episode_steps=None):
    agent_spec = AgentSpec(
        interface=AgentInterface.from_type(
            AgentType.LanerWithSpeed, max_episode_steps=max_episode_steps
        ),
        agent_builder=ChaseWaypointsAgent,
    )

    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=scenarios,
        agent_specs={"SingleAgent": agent_spec},
        headless=headless,
        sumo_headless=True,
    )

    # Convert SMARTS observations to standardized gym-compliant observations.
    env = StandardObs(env=env)

    # Convert `env.step()` and `env.reset()` from multi-agent interface to
    # single-agent interface.
    env = SingleAgent(env=env)

    for episode in episodes(n=num_episodes):
        agent = agent_spec.build_agent()
        observation = env.reset()
        episode.record_scenario(env.scenario_log)

        done = False
        while not done:
            agent_action = agent.act(observation)
            observation, reward, done, info = env.step(agent_action)
            episode.record_step(observation, reward, done, info)

    env.close()


def _build_scenario(scenario: List[str]):
    build_scenario = " ".join(["scl scenario build-all --clean"] + scenario)
    subprocess.call(build_scenario, shell=True)


if __name__ == "__main__":
    parser = default_argument_parser("single-agent-example")
    args = parser.parse_args()

    if not args.scenarios:
        args.scenarios = [
            str(pathlib.Path(__file__).absolute().parents[1] / "scenarios" / "loop")
        ]

    _build_scenario(args.scenarios)

    main(
        scenarios=args.scenarios,
        headless=args.headless,
        num_episodes=args.episodes,
    )
