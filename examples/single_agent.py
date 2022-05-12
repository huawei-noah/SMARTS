import logging
import pathlib

import gym
from functools import partial
from examples.argument_parser import default_argument_parser
from smarts.core.agent import Agent
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.sensors import Observation
from smarts.core.utils.episodes import episodes
from smarts.env.wrappers.single_agent import SingleAgent
from smarts.sstudio import build_scenario
from smarts.zoo.agent_spec import AgentSpec

logging.basicConfig(level=logging.INFO)


class TargetLaneAgent(Agent):
    def act(self, obs: Observation):
        return (
            obs.waypoint_paths[0][0].speed_limit,
            self._target_lane - obs.ego_vehicle_state.lane_index,
        )

    def __init__(self, target_lane):
        self._target_lane = target_lane


def main(scenarios, headless, num_episodes, max_episode_steps=None):
    agent_spec = AgentSpec(
        interface=AgentInterface.from_type(
            AgentType.LanerWithSpeed, max_episode_steps=max_episode_steps
        ),
        agent_builder=partial(TargetLaneAgent, 3),
    )

    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=scenarios,
        agent_specs={"SingleAgent": agent_spec},
        headless=headless,
        sumo_headless=True,
    )

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


if __name__ == "__main__":
    parser = default_argument_parser("single-agent-example")
    args = parser.parse_args()

    if not args.scenarios:
        args.scenarios = [
            str(pathlib.Path(__file__).absolute().parents[1] / "scenarios" / "loop")
        ]

    build_scenario(args.scenarios)

    main(
        scenarios=args.scenarios,
        headless=args.headless,
        num_episodes=args.episodes,
    )
