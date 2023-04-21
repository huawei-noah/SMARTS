import sys
from pathlib import Path

import gymnasium as gym

sys.path.insert(0, str(Path(__file__).parents[2].absolute()))
from examples.tools.argument_parser import default_argument_parser
from smarts.core.agent_interface import AgentInterface
from smarts.core.controllers import ActionSpaceType
from smarts.sstudio.scenario_construction import build_scenarios


def main(scenarios, headless, num_episodes):

    interfaces = {
        agent_name: AgentInterface(
            action=ActionSpaceType.Continuous,
            drivable_area_grid_map=False,
            lane_positions=True,
            lidar_point_cloud=False,
            occupancy_grid_map=False,
            road_waypoints=False,
            signals=False,
            top_down_rgb=False,
        )
        for agent_name in ["A1", "A2", "A3", "A4"]
    }

    env = gym.make(
        "smarts.env:hiway-v1",
        scenarios=scenarios,
        agent_interfaces=interfaces,
    )

    for episode in range(num_episodes):
        obs, _ = env.reset()
        print(obs.keys())
        assert len(obs.keys()) == 4
        terminated = {"__all__": False}
        while not terminated["__all__"]:
            actions = {
                agent_id: (0, 1, 0)  # Break at all times
                for agent_id, agent_obs in obs.items()
            }
            obs, rewards, terminated, truncated, infos = env.step(actions)
            break

            # ego_pos = obs["Agent_0"]["ego_vehicle_state"]["position"]
            # ego_event = obs["Agent_0"]["events"]
            # leader_pos = env.smarts.vehicle_index.vehicle_position("Leader-007")
            # print(f"Ego pos: {ego_pos}")
            # print(f"Ego event: {ego_event}")
            # print(f"Leader pos: {leader_pos}")
            # print("\n")

        # print("\n")
        # print(f"Episode steps: {obs['Agent_0']['steps_completed']}")
        # assert obs["Agent_0"]["steps_completed"] > 100
        # print("\n\n")
        print("----------------------------------------------")

    env.close()


if __name__ == "__main__":
    parser = default_argument_parser("laner")
    args = parser.parse_args()

    if not args.scenarios:
        # args.scenarios = [str(
        #         Path(__file__).absolute().parents[2]
        #         / "scenarios"
        #         / "sumo"
        #         / "platoon"
        #         / "straight_2lane_sumo_t_agents_1"
        #     )
        # ]
        args.scenarios = [
            str(Path(__file__).absolute().parents[2] / "scenarios" / "sumo" / "loop")
        ]

    build_scenarios(scenarios=args.scenarios)

    main(
        scenarios=args.scenarios,
        headless=args.headless,
        num_episodes=100,
    )
