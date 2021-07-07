import logging

import gym

from examples.argument_parser import default_argument_parser
from smarts.core.agent import Agent, AgentSpec
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.sensors import Observation
from smarts.core.utils.episodes import episodes

logging.basicConfig(level=logging.INFO)

AGENT_ID = "Agent-007"


class CheckLaneOrderAgent(Agent):
    def act(self, obs: Observation):
        longest_val = 0
        longest_index = 0
        for i in range(len(obs.waypoint_paths)):
            wpp = obs.waypoint_paths[i]
            # print(f"End path is: {wpp[-1].lane_id}")
            if len(wpp) > longest_val:
                longest_val=len(wpp)
                longest_index=i
        return ( 
            1, 
            1 if longest_index > obs.ego_vehicle_state.lane_index 
            else (-1 if longest_index < obs.ego_vehicle_state.lane_index else 0)
        )


def main(scenarios, sim_name, headless, num_episodes, seed, max_episode_steps=None):
    agent_spec = AgentSpec(
        interface=AgentInterface.from_type(
            AgentType.LanerWithSpeed, max_episode_steps=max_episode_steps
        ),
        agent_builder=CheckLaneOrderAgent,
    )

    agent_spec.interface.max_episode_steps = 2

    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=scenarios,
        agent_specs={AGENT_ID: agent_spec},
        sim_name=sim_name,
        headless=True,
        visdom=False,
        timestep_sec=0.1,
        sumo_headless=True, #False,
        seed=42,
        # zoo_addrs=[("10.193.241.236", 7432)], # Sample server address (ip, port), to distribute social agents in remote server.
        # envision_record_data_replay_path="./data_replay",
    )

    num_episodes=1
    for episode in episodes(n=num_episodes):
        agent = agent_spec.build_agent()
        print(" RESET VVV 0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000")
        observations = env.reset()
        episode.record_scenario(env.scenario_log)

        dones = {"__all__": False}
        while not dones["__all__"]:
            print("AGENT OBS VVV 0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000")
            agent_obs = observations[AGENT_ID]

            print("----------------------------------------------")    
            wp_test = agent_obs.waypoint_paths
            print(f"Len of wp:{len(wp_test)}")
            for index in range(len(wp_test)):
                print(f"Path:{index}, Len:{len(wp_test[index])}, LaneID:{wp_test[index][-1].lane_id}")
            print("----------------------------------------------")    

            agent_action = agent.act(agent_obs)
            observations, rewards, dones, infos = env.step({AGENT_ID: agent_action})
            episode.record_step(observations, rewards, dones, infos)

    env.close()


if __name__ == "__main__":
    parser = default_argument_parser("single-agent-example")
    args = parser.parse_args()

    main(
        scenarios=args.scenarios,
        sim_name=args.sim_name,
        headless=args.headless,
        num_episodes=args.episodes,
        seed=args.seed,
    )
