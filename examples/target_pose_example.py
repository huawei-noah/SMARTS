import logging
import pathlib

import gym

from examples.argument_parser import default_argument_parser
from smarts.core.agent import Agent
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.controllers import ActionSpaceType
from smarts.core.sensors import Observation
from smarts.core.utils.episodes import episodes
from smarts.env.wrappers.single_agent import SingleAgent
from smarts.sstudio import build_scenario
from smarts.zoo.agent_spec import AgentSpec

logging.basicConfig(level=logging.INFO)

import numpy as np


class NonInteractiveAgent(Agent):
    def __init__(self, speed=5, target_lane_index=None):
        self.speed = speed
        if target_lane_index is None:
            target_lane_index = {}
        self.target_lanes = [
            f"{edge}_{lane_index}" for edge, lane_index in target_lane_index.items()
        ]

    def act(self, obs: Observation):
        # Waypoint searching approach:
        # 1. Use the first waypoint path as default
        # 2. Look for current waypoint path
        # 3. Look for a waypoint path in the target lanes
        current_lane_id = obs.ego_vehicle_state.lane_id
        wp = obs.road_waypoints.lanes[current_lane_id][0][:5][-1]
        # print("a"+wp.lane_id)
        for waypoints in obs.road_waypoints.lanes.values():
            for lane_waypoints in waypoints:
                if lane_waypoints[0].lane_id == current_lane_id:
                    wp = lane_waypoints[:5][-1]
                    # print("b"+wp.lane_id)
                    break
        for target in reversed(self.target_lanes):
            tl = obs.road_waypoints.lanes.get(target, None)
            if tl is None:
                continue
            for waypoints in tl:
                print(target + " but actually: "+waypoints[0].lane_id)

                if waypoints[:5][-1].lane_id in self.target_lanes:
                    wp = waypoints[:5][-1]
        dist_to_wp = wp.dist_to(obs.ego_vehicle_state.position)
        print(f"selected {wp.lane_id} {dist_to_wp / self.speed}")
        return np.array([*wp.pos, wp.heading, dist_to_wp / self.speed])


def main(scenarios, headless, num_episodes, max_episode_steps=None):
    agent_spec = AgentSpec(
        interface=AgentInterface(waypoints=False, road_waypoints=True, action=ActionSpaceType.TargetPose, max_episode_steps=max_episode_steps),
        agent_builder=NonInteractiveAgent,
        agent_params={"target_lane_index": {
            # "E3-35": 1,
            # "E3-3l": 1,
            ":J3_19": 0,
            "E3-3s": 1
        }},
    )

    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=scenarios,
        agent_specs={"SingleAgent": agent_spec},
        headless=headless,
        sumo_headless=False,
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
            str(pathlib.Path(__file__).absolute().parents[1] / "scenarios" / "cross")
        ]

    build_scenario(args.scenarios)

    main(
        scenarios=args.scenarios,
        headless=args.headless,
        num_episodes=2,
        max_episode_steps=800
    )
