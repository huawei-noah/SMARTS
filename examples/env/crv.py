import os
import sys
from pathlib import Path

import gymnasium as gym
from gymnasium.wrappers.record_video import RecordVideo

from smarts.core.utils.episodes import Episode, episode_range, episodes
from smarts.env.gymnasium.wrappers.episode_logger import EpisodeLogger
from smarts.zoo import registry

SMARTS_DIR = Path(os.path.abspath(""))
sys.path.insert(0, SMARTS_DIR)

from examples.env import figure_eight_env

env = gym.make("figure_eight-v0", disable_env_checker=True)
env: gym.Env = EpisodeLogger(env)

import zoo.policies.keep_lane_agent

agent = registry.make_agent("zoo.policies:keep-lane-agent-v0")

for episode in episode_range(max_steps=450):
    observation, info = env.reset()
    reward, terminated, truncated, info = None, False, False, None
    while episode.continues(observation, reward, terminated, truncated, info):
        action = agent.act(observation)
        observation, reward, _, terminated, info = env.step(action)

env.close()
