import sys
from pathlib import Path

import gym

sys.path.insert(0, str(Path(__file__).parents[1]))
from tools.argument_parser import default_argument_parser

from smarts import sstudio
from smarts.core.agent import Agent
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.sensors import Observation
from smarts.core.utils.episodes import episodes
from smarts.env.hiway_env_v1 import HiWayEnvV1
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
    agent_interface = AgentInterface.from_type(
        AgentType.LanerWithSpeed, max_episode_steps=max_episode_steps
    )

    env = HiWayEnvV1(
        scenarios=scenarios,
        agent_interfaces={"SingleAgent": agent_interface},
        headless=headless,
    )

    for episode in episodes(n=num_episodes):
        agent = ChaseViaPointsAgent()
        observation, info = env.reset()
        episode.record_scenario(env.scenario_log)

        done = False
        while not done:
            agent_action = agent.act(observation["SingleAgent"])
            observation, reward, terminated, truncated, info = env.step(
                {"SingleAgent": agent_action}
            )
            done = truncated or terminated
            episode.record_step(observation, reward, {"__all__": done}, info)

    env.close()


if __name__ == "__main__":
    parser = default_argument_parser("single-agent-example")
    args = parser.parse_args()

    if not args.scenarios:
        args.scenarios = [
            str(Path(__file__).absolute().parents[2] / "scenarios" / "sumo" / "loop")
        ]

    sstudio.build_scenario(scenario=args.scenarios)

    main(
        scenarios=args.scenarios,
        headless=args.headless,
        num_episodes=args.episodes,
    )