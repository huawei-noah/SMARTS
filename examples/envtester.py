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
        interface=AgentInterface.from_type(
            AgentType.Laner, max_episode_steps=max_episode_steps
        ),
        agent_builder=lambda: Agent.from_function(lambda _: 0),
    )

    # Action map for AgentType.Laner
    # action_map = {
    #     0: "keep_lane",
    #     1: "slow_down",
    #     2: "change_lane_left",
    #     3: "change_lane_right",
    # }

    env = gym.make(
        "smarts.env:merge-v0",
        headless=True,
        sumo_headless=False,
        action_space="Lane",
    )

    for episode in episodes(n=num_episodes):
        agent = agent_spec.build_agent()
        observation = env.reset()

        done = False
        while not done:

            agent_action = agent.act(observation)
            observation, reward, done, info = env.step(agent_action)
            if observation["events"]["reached_goal"]:
                print("HURRAY REACHED GOAL !!!")

    env.close()


if __name__ == "__main__":
    parser = default_argument_parser("test-example")
    args = parser.parse_args()

    main(
        num_episodes=30,
    )
