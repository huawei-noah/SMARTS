import gymnasium as gym
from smarts.env.wrappers.metrics import Metrics

env = gym.make(
    "smarts.env:driving-smarts-competition-v0", scenario="1_to_2lane_left_turn_c"
)
# TODO: this should be made to work
# env = Metrics(env)

observation, info = env.reset(seed=42)
for _ in range(1000):
    # Generally [x-coordinate, y-coordinate, heading]
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    assert env.observation_space.contains(observation)

    if terminated or truncated:
        observation, info = env.reset()

print(env.score())
env.close()
