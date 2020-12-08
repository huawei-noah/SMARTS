import logging

import gym

from smarts.core.utils.episodes import episodes
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.agent import AgentSpec, Agent
from smarts.core.sensors import Observation

from examples import default_argument_parser


logging.basicConfig(level=logging.INFO)

AGENT_ID = "Agent-007"


class ChaseViaPointsAgent(Agent):
    def act(self, obs: Observation):
        if (
            len(obs.via_data.near_via_points) < 1
            or obs.ego_vehicle_state.edge_id != obs.via_data.near_via_points[0].edge_id
        ):
            return "keep_lane"

        nearest = obs.via_data.near_via_points[0]
        if nearest.lane_index == obs.ego_vehicle_state.lane_index:
            speed_dif = obs.ego_vehicle_state.speed - nearest.required_speed
            return "slow_down" if speed_dif > 1 else "keep_lane"

        return (
            "change_lane_left"
            if nearest.lane_index > obs.ego_vehicle_state.lane_index
            else "change_lane_right"
        )


def main(
    scenarios, headless, num_episodes, seed, auth_key=None, max_episode_steps=None
):
    agent_spec = AgentSpec(
        interface=AgentInterface.from_type(
            AgentType.Laner, max_episode_steps=max_episode_steps
        ),
        agent_builder=ChaseViaPointsAgent,
    )

    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=scenarios,
        agent_specs={AGENT_ID: agent_spec},
        headless=headless,
        visdom=False,
        timestep_sec=0.1,
        sumo_headless=True,
        seed=seed,
        # zoo_workers=[("143.110.210.157", 7432)], # Distribute social agents across these workers
        auth_key=auth_key,
        # envision_record_data_replay_path="./data_replay",
    )

    for episode in episodes(n=num_episodes):
        agent = agent_spec.build_agent()
        observations = env.reset()
        episode.record_scenario(env.scenario_log)

        dones = {"__all__": False}
        while not dones["__all__"]:
            agent_obs = observations[AGENT_ID]
            agent_action = agent.act(agent_obs)
            observations, rewards, dones, infos = env.step({AGENT_ID: agent_action})
            episode.record_step(observations, rewards, dones, infos)

    env.close()


if __name__ == "__main__":
    parser = default_argument_parser("single-agent-example")
    parser.add_argument(
        "--auth_key",
        type=str,
        default=None,
        help="Authentication key for connection to run agent",
    )
    args = parser.parse_args()
    auth_key = args.auth_key if args.auth_key else ""

    main(
        scenarios=args.scenarios,
        headless=args.headless,
        num_episodes=args.episodes,
        seed=args.seed,
        auth_key=auth_key,
    )
