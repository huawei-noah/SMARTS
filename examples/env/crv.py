import gymnasium as gym

from smarts.zoo import registry
from smarts.env.gymnasium.wrappers.episode_logger import EpisodeLogger
from smarts.core.utils.episodes import episode_range
from smarts.env.wrappers.record_video import RecordVideo
from smarts.core.utils.import_utils import import_module_from_file

import sys
import os
from pathlib import Path

sys.path.insert(0, Path(os.path.abspath("")))
print(Path(os.path.abspath("")))

from examples.env import figure_eight_env

env = gym.make("figure_eight-v0", disable_env_checker=True)
env: gym.Env = RecordVideo(
    env, video_folder="videos", video_length=40, step_trigger=lambda s: s % 100 == 0
)
env: gym.Env = EpisodeLogger(env)

import zoo.policies.keep_lane_agent

agent = registry.make_agent("zoo.policies:keep-lane-agent-v0")

for episode in episode_range(max_steps=450):
    observation = env.reset()
    reward, done, info = None, False, None
    while episode.continues(observation, reward, done, info):
        action = agent.act(observation)
        observation, reward, _, done, info = env.step(action)

env.close()