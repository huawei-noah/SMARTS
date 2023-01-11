import gymnasium as gym
import random
from smarts.core.agent import Agent

from smarts.core.agent import Agent
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.env.wrappers.single_agent import SingleAgent


class RandomAgent(Agent):
    def act(self, obs):
        return (random.randrange(0, 60), random.randrange(-1, 2))


gym.register(
    id="test_env-v0",
    entry_point="smarts.env.test_env:test_env",
)


def entrypoint():
    agent_interface = AgentInterface.from_type(
        AgentType.LanerWithSpeed, max_episode_steps=20
    )
    env = gym.make(
        "smarts.env:test_env-v0", agent_interfaces={"SingleAgent": agent_interface}
    )
    env = SingleAgent(env)

    return env


gym.register("single_agent-v0", entrypoint)

agent = RandomAgent()
env = gym.make("single_agent-v0")

observation, info = env.reset()
for _ in range(100):
    # Generally [x-coordinate, y-coordinate, heading]
    action = agent.act(observation)
    assert env.action_space.contains(action)
    observation, reward, terminated, truncated, info = env.step(action)
    assert env.observation_space.contains(observation)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
