import sys
sys.path.insert(0, "./examples/env")
sys.path.insert(0, "./zoo")
import baseline
import policies.keep_lane_agent

import gym

from smarts.zoo import registry
from smarts.env.wrappers.episode_logger import EpisodeLogger
from smarts.env.wrappers.record import RecordVideo, RenderVideo
from smarts.core.utils.episodes import episode_range

#FormatObs should already be applied
env = gym.make("figure_eight-v0")
# gym.wrappers.Monitor
env: gym.Env = RecordVideo(env, frequency=10)
env: gym.Env = RenderVideo(env)
env: gym.Env = EpisodeLogger(env)

agent = registry.make_agent("zoo.policies:keep-lane-agent-v0")
for episode in episode_range(max_steps=450):
    observation = env.reset()
    reward, done, info = None, False, None
    while episode.register_step(observation, reward, done, info):
        action = agent.act(observation)
        observation, reward, done, info = env.step(action)

env.close()