from typing import Dict

import gym
import numpy as np
from merge.util import plotter3d


class RGB(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=env.observation_space["rgb"].shape,
            dtype=np.uint8,
        )

    def observation(self, obs: Dict[str, gym.Space]) -> np.ndarray:
        rgb = obs["rgb"]

        # Ego vehicle is 1.5mx3.75m
        # Road width = 6.25m

        # Repaint ego
        # clr = (122, 140, 153)
        # repainted = img.copy()
        # repainted[120:135, 125:131, 0] = clr[0]
        # repainted[120:135, 125:131, 1] = clr[1]
        # repainted[120:135, 125:131, 2] = clr[2]

        # RGB to grayscale
        # R, G, B = repainted[:, :, 0], repainted[:, :, 1], repainted[:, :, 2]
        # gray = 0.2989 * R + 0.587 * G + 0.114 * B

        # Expand dims
        # expanded = np.expand_dims(gray, -1)

        # Channel first
        # rgb = rgb.transpose(2, 0, 1)

        # Resize image to 64x64
        # resized = transposed[:, 96:160, 96:160]

        # rep = repainted.transpose(2, 0, 1)
        # plotter(rgb, 3, name="from obs")

        # intersection_util.plotter3d(rgb, rgb_gray=3, name="From Obs")

        return np.uint8(rgb)


class Concatenate(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=old_space.shape[1:-1] + (old_space.shape[0] * old_space.shape[-1],),
            dtype=np.uint8,
        )

    def observation(self, obs):
        # print("Before:",obs.shape)
        # plotter3d(obs,rgb_gray=3,channel_order="last",name="before")
        assert len(obs.shape) == 4
        obs = np.concatenate(obs[:], axis=-1)
        # print("After:", obs.shape)
        # plotter3d(obs, rgb_gray=3,channel_order="last",name="after",pause=0)
        return obs
