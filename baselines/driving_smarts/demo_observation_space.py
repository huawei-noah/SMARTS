from pprint import pprint as print

import gymnasium as gym

env = gym.make(
    "smarts.env:driving-smarts-competition-v0",
    scenario="3lane_merge_multi_agent",
    sumo_headless=False,
)

print("Action space:")
print(env.action_space)
print("")
print("Observation space:")
print(env.observation_space)

env.close()
