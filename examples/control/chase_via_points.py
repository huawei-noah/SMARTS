import sys
from pathlib import Path

import gymnasium as gym

sys.path.insert(0, str(Path(__file__).parents[2].absolute()))
from examples.tools.argument_parser import default_argument_parser
from smarts.core.agent import Agent
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.observations import Observation
from smarts.core.utils.episodes import episodes
from smarts.env.utils.observation_conversion import ObservationOptions
from smarts.sstudio.scenario_construction import build_scenarios

N_AGENTS = 3
AGENT_IDS = ["Agent_%i" % i for i in range(N_AGENTS)]


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
    agent_interfaces = {
        agent_id: AgentInterface.from_type(
            AgentType.LanerWithSpeed,
            max_episode_steps=max_episode_steps,
        )
        for agent_id in AGENT_IDS
    }

    env = gym.make(
        "smarts.env:hiway-v1",
        scenarios=scenarios,
        agent_interfaces=agent_interfaces,
        headless=headless,
        observation_options=ObservationOptions.unformatted,
    )

    for episode in episodes(n=num_episodes):
        agents = {
            agent_id: ChaseViaPointsAgent() for agent_id in agent_interfaces.keys()
        }
        observations, _ = env.reset()
        episode.record_scenario(env.scenario_log)

        terminateds = {"__all__": False}
        while not terminateds["__all__"]:
            actions = {
                agent_id: agents[agent_id].act(agent_obs)
                for agent_id, agent_obs in observations.items()
            }
            observations, rewards, terminateds, truncateds, infos = env.step(actions)
            episode.record_step(observations, rewards, terminateds, truncateds, infos)

    env.close()


if __name__ == "__main__":
    parser = default_argument_parser("chase-via-points")
    args = parser.parse_args()

    if not args.scenarios:
        args.scenarios = [
            str(Path(__file__).absolute().parents[2] / "scenarios" / "sumo" / "loop")
        ]

    build_scenarios(scenarios=args.scenarios)

    main(
        scenarios=args.scenarios,
        headless=args.headless,
        num_episodes=args.episodes,
    )
