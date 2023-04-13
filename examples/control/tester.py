import sys
from pathlib import Path

import gymnasium as gym

sys.path.insert(0, str(Path(__file__).parents[2].absolute()))
from examples.tools.argument_parser import default_argument_parser
from smarts.core.agent_interface import AgentInterface
from smarts.sstudio.scenario_construction import build_scenarios

def main(scenario, headless, num_episodes, max_episode_steps=None):
    from smarts.core.controllers import ActionSpaceType
    interface = AgentInterface(
        action=ActionSpaceType.Continuous,
        drivable_area_grid_map=False,
        lane_positions=True,
        lidar_point_cloud=False,
        occupancy_grid_map=False,
        road_waypoints=False,
        signals=False,
        top_down_rgb=False,
    )

    env = gym.make(
        "smarts.env:platoon-v0",
        scenario=scenario,
        agent_interface=interface,
        headless=headless,
        sumo_headless=False,
    )

    for episode in range(num_episodes):
        obs, _ = env.reset()
        dones = {"__all__": False}
        while not dones["__all__"]:
            actions = {
                agent_id: (0,1,0) # Break at all times
                for agent_id, agent_obs in obs.items()
            }
            obs, rewards, terminated, truncated, infos = env.step(actions)
            seen = obs["Agent_0"]["ego_vehicle_state"]["position"]
            e = obs["Agent_0"]["events"]
            print(seen, e)

    env.close()


if __name__ == "__main__":
    parser = default_argument_parser("laner")
    args = parser.parse_args()

    if not args.scenarios:
        args.scenarios = "/home/kyber/workspace/SMARTS/scenarios/sumo/platoon/straight_2lane_sumo_t_agents_1"

    build_scenarios(scenarios=[args.scenarios])

    main(
        scenario=args.scenarios,
        headless=args.headless,
        num_episodes=10,
        max_episode_steps=args.max_episode_steps,
    )
