import gymnasium as gym
from smarts.env.wrappers.metrics import CompetitionMetrics, CompetitionMetrics

env = gym.make(
    "smarts.env:driving-smarts-competition-v0",
    scenario="3lane_merge_multi_agent",
    sumo_headless=False,
)
env = CompetitionMetrics(env)

observation, info = env.reset(seed=42)
for _ in range(30):
    # Generally [x-coordinate, y-coordinate, heading]
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    env.observation_space.contains(observation)

    if terminated or truncated:
        observation, info = env.reset()

print(env.score())
env.close()
