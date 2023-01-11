from typing import Dict

import gym
import numpy as np


class Observation(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        old_space = env.observation_space["top_down_rgb"]
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(old_space.shape[-1],) + old_space.shape[:-1],
            dtype=np.uint8,
        )

    def observation(self, obs: Dict[str, gym.Space]) -> np.ndarray:
        top_down_rgb = obs["top_down_rgb"]

        # Channel first
        top_down_rgb = top_down_rgb.transpose(2, 0, 1)

        return np.uint8(top_down_rgb)
