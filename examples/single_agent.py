import logging
import os
import pathlib

import gym
import numpy as np

from smarts.core.agent import Agent, AgentSpec
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.utils.episodes import episodes
from smarts.env.wrappers.single_agent import SingleAgent
from smarts.env.wrappers.standard_obs import StandardObs, StdObs

# The following ugliness was made necessary because the `aiohttp` #
# dependency has an "examples" module too.  (See PR #1120.)
if __name__ == "__main__":
    from argument_parser import default_argument_parser
else:
    from .argument_parser import default_argument_parser

logging.basicConfig(level=logging.INFO)


class ChaseWaypointsAgent(Agent):
    def act(self, obs: StdObs):
        cur_lane_index = obs.ego_vehicle_state["lane_index"]
        next_lane_index = obs.waypoint_paths["lane_index"][0, 0]

        return (
            obs.waypoint_paths["speed_limit"][0, 0] / 2,
            np.sign(next_lane_index - cur_lane_index),
        )


def main(scenarios, headless, num_episodes, max_episode_steps=100):
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
        visdom=False,
        sumo_headless=True,
        envision_record_data_replay_path=None,
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


def _build_scenario(scenario):
    build_scenario = f"scl scenario build-all --clean {scenario}"
    os.system(build_scenario)


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
