import gymnasium as gym
from smarts.env.wrappers.metrics import Metrics, CompetitionMetrics

env = gym.make(
    "smarts.env:driving-smarts-competition-v0", scenario="3lane_merge_multi_agent"
)
# TODO: this should be made to work
env = CompetitionMetrics(env)

observation, info = env.reset(seed=42)
for _ in range(1000):
    # Generally [x-coordinate, y-coordinate, heading]
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    # assert env.observation_space.contains(observation)

    if terminated or truncated:
        observation, info = env.reset()

print(env.score())
env.close()


# import gym
# from smarts.env.wrappers.metrics import Metrics

# env = gym.make("smarts.env:multi-scenario-v0", scenario="3lane_merge_multi_agent")
# # TODO: this should be made to work
# env = Metrics(env)

# observation = env.reset()
# for _ in range(1000):
#     # Generally [x-coordinate, y-coordinate, heading, 0.1]
#     action = env.action_space.sample()
#     observation, reward, truncated, info = env.step(action)
#     # assert env.observation_space.contains(observation)

#     if truncated:
#         observation = env.reset()

# print(env.score())
# env.close()
