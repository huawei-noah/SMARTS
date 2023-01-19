import gymnasium as gym

from smarts.env.gymnasium.wrappers.metrics import CompetitionMetrics

env = gym.make(
    "smarts.env:driving-smarts-competition-v0",
    scenario="3lane_merge_multi_agent",
    sumo_headless=False,
)
env = CompetitionMetrics(env)

observation, info = env.reset(seed=42)
MAX_RESETS = 5
current_resets = 0
while current_resets < MAX_RESETS:
    # Generally [x-coordinate, y-coordinate, heading]
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    env.observation_space.contains(observation)

    if terminated or truncated:
        current_resets += 1
        observation, info = env.reset()

print("\n".join(f"- {k}: {v}" for k, v in env.score().items()))
env.close()
