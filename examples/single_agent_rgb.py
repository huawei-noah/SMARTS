import logging

import gym

# import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from pathlib import Path

from examples.argument_parser import default_argument_parser
from smarts.core.agent import Agent, AgentSpec
from smarts.core.agent_interface import AgentInterface, RGB
from smarts.core.controllers import ActionSpaceType
from smarts.core.sensors import Observation
from smarts.core.utils.episodes import episodes
from smarts.core.colors import Colors

logging.basicConfig(level=logging.INFO)

AGENT_ID = "Agent-007"


class ChaseViaPointsAgent(Agent):
    def act(self, obs: Observation):
        if (
            len(obs.via_data.near_via_points) < 1
            or obs.ego_vehicle_state.edge_id != obs.via_data.near_via_points[0].edge_id
        ):
            return (obs.waypoint_paths[0][0].speed_limit, 0)

        nearest = obs.via_data.near_via_points[0]
        if nearest.lane_index == obs.ego_vehicle_state.lane_index:
            return (nearest.required_speed, 0)

        return (
            nearest.required_speed,
            1 if nearest.lane_index > obs.ego_vehicle_state.lane_index else -1,
        )


def main(scenarios, sim_name, headless, num_episodes, seed, max_episode_steps=None):
    agent_spec = AgentSpec(
        interface=AgentInterface(
            waypoints=True,
            rgb=RGB(),
            action=ActionSpaceType.LaneWithContinuousSpeed,
        ),
        agent_builder=ChaseViaPointsAgent,
    )

    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=scenarios,
        agent_specs={AGENT_ID: agent_spec},
        sim_name=sim_name,
        headless=headless,
        visdom=False,
        timestep_sec=0.1,
        sumo_headless=True,
        seed=seed,
        # zoo_addrs=[("10.193.241.236", 7432)], # Sample server address (ip, port), to distribute social agents in remote server.
        # envision_record_data_replay_path="./data_replay",
    )

    out_filename = Path(__file__).parent.resolve().joinpath("results", "results.txt")
    for episode in episodes(n=num_episodes):
        agent = agent_spec.build_agent()
        observations = env.reset()
        episode.record_scenario(env.scenario_log)

        dones = {"__all__": False}
        step = 0
        while not dones["__all__"]:
            agent_obs = observations[AGENT_ID]

            # Plot graph
            step += 1
            if step % 100 == 0:
                rgb = agent_obs.top_down_rgb.data
                line1 = str(rgb.shape)
                print(line1)
                pixel = rgb[128, 128, :] / 255
                des = np.array(Colors.Red.value)
                line2 = "Center pixel match: " + str(np.allclose(pixel, des[:-1], 1e-4))
                print(line2)
                print("--------------------")
                with open(out_filename, "a") as out_file:
                    out_file.write(line1 + "\n")
                    out_file.write(line2 + "\n")
                    now = datetime.now()
                    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
                    out_file.write(dt_string + "\n")
                    out_file.write("--------------------\n")

            # fig=plt.figure(figsize=(10,10))
            # img = agent_obs.top_down_rgb.data
            # fig.add_subplot(1, 1, 1)
            # plt.imshow(img)
            # plt.show()

            agent_action = agent.act(agent_obs)
            observations, rewards, dones, infos = env.step({AGENT_ID: agent_action})
            episode.record_step(observations, rewards, dones, infos)

    env.close()


if __name__ == "__main__":
    parser = default_argument_parser("single-agent-example")
    args = parser.parse_args()

    main(
        scenarios=["./scenarios/loop/"],
        sim_name=args.sim_name,
        headless=True,
        num_episodes=2,
        seed=args.seed,
    )
