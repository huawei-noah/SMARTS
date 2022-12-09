import logging
import pathlib

import gym
from argument_parser import default_argument_parser

from smarts import sstudio
from smarts.core.agent import Agent
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.sensors import Observation
from smarts.core.utils.episodes import episodes
from smarts.env.wrappers.single_agent import SingleAgent
from smarts.zoo.agent_spec import AgentSpec


class ChaseViaPointsAgent(Agent):
    def act(self, obs: Observation):
        if (
            len(obs.via_data.near_via_points) < 1
            or obs.ego_vehicle_state.road_id != obs.via_data.near_via_points[0].road_id
        ):
            return (obs.waypoint_paths[0][0].speed_limit, 0)

        nearest = obs.via_data.near_via_points[0]
        if nearest.lane_index == obs.ego_vehicle_state.lane_index:
            return (nearest.required_speed, 0)

        return (
            nearest.required_speed,
            1 if nearest.lane_index > obs.ego_vehicle_state.lane_index else -1,
        )


def main(scenarios, headless, num_episodes, max_episode_steps=None):
    agent_spec = AgentSpec(
        interface=AgentInterface.from_type(
            AgentType.LanerWithSpeed,
            max_episode_steps=800,
            accelerometer=True,
            neighborhood_vehicles=True,
            road_waypoints=True,
            waypoints=True,
        ),
        agent_builder=ChaseViaPointsAgent,
    )

    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=scenarios,
        agent_specs={"SingleAgent": agent_spec},
        headless=True,
        sumo_headless=False,
    )

    from smarts.env.wrappers.metrics import Metrics

    env = Metrics(env=env)

    # Convert `env.step()` and `env.reset()` from multi-agent interface to
    # single-agent interface.
    env = SingleAgent(env=env)

    for episode in episodes(n=100):
        agent = agent_spec.build_agent()
        observation = env.reset()
        episode.record_scenario(env.scenario_log)

        done = False
        while not done:
            agent_action = agent.act(observation)
            observation, reward, done, info = env.step(agent_action)
            episode.record_step(observation, reward, done, info)

        print(f"{episode.index} score == {env.score()}")
        print(f"{episode.index} records == {env.records()}")

    env.close()


if __name__ == "__main__":
    parser = default_argument_parser("single-agent-example")
    args = parser.parse_args()

    if not args.scenarios:
        args.scenarios = [
            str(
                pathlib.Path(__file__).absolute().parents[1]
                / "smarts"
                / "scenarios"
                / "intersection"
                / "1_to_1lane_left_turn_c"
            )
        ]

    sstudio.build_scenario(scenario=args.scenarios)

    main(
        scenarios=args.scenarios,
        headless=args.headless,
        num_episodes=args.episodes,
    )
