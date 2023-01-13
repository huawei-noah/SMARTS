import gymnasium as gym

from smarts.env.gymnasium.wrappers.episode_limit import EpisodeLimit
from smarts.env.gymnasium.wrappers.metrics import CompetitionMetrics

env = gym.make(
    "smarts.env:driving-smarts-competition-v0",
    scenario="3lane_merge_multi_agent",
    sumo_headless=False,
)
env = CompetitionMetrics(env)
env = EpisodeLimit(env, 5)

observation, info = env.reset(seed=42)
while not info.get("reached_episode_limit"):
    # Generally [x-coordinate, y-coordinate, heading]
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    env.observation_space.contains(observation)

    if terminated or truncated:
        observation, info = env.reset()

print("\n".join(f"- {k}: {v}" for k, v in env.score().items()))
env.close()
