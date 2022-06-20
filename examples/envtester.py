import logging

import gym

from examples.argument_parser import default_argument_parser
from smarts.core.agent import Agent
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.utils.episodes import episodes
from smarts.zoo.agent_spec import AgentSpec

logging.basicConfig(level=logging.INFO)


def main(num_episodes, max_episode_steps=50):
    agent_spec = AgentSpec(
        # interface=AgentInterface.from_type(
        #     AgentType.Laner, max_episode_steps=max_episode_steps
        # ),
        agent_builder=lambda: Agent.from_function(lambda _: "keep_lane"),
    )

    env = gym.make(
        "smarts.env:multi-scenario-v0",
        scenario="3lane_overtake",
        headless=True,
        sumo_headless=True,
        action_space="Lane",
    )


    def modify_info(obs, reward, info):
        info["security"] = "checked----"
        return info 

    for agent_name, env_agent_spec in env.agent_specs.items():
        env_agent_spec.info_adapter = modify_info




    for episode in episodes(n=num_episodes):
        observations = env.reset()
        agents = {
            agent_id: agent_spec.build_agent() for agent_id in env.agent_specs.keys()
        }

        dones = {"__all__": False}
        while not dones["__all__"]:
            actions = {
                agent_id: agents[agent_id].act(agent_obs)
                for agent_id, agent_obs in observations.items()
            }
            observations, rewards, dones, infos = env.step(actions)
            # import time
            # time.sleep(0.1)

            print(infos["Agent_0"]["security"])

            for agent_id, agent_obs in observations.items():
                if agent_obs.events.reached_goal:
                    print(f"{agent_id} : HURRAY REACHED GOAL !!!")
                if agent_obs.events.collisions:
                    print(f"{agent_id} : collided !!!")
    env.close()


if __name__ == "__main__":
    parser = default_argument_parser("test-example")
    args = parser.parse_args()

    main(
        num_episodes=30,
    )
