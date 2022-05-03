import gym
import numpy as np
from sb3 import util as sb3_util

from smarts.env.wrappers import format_obs


class ObservationCNN(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        old_space = env.observation_space["rgb"]
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(old_space.shape[-1],) + old_space.shape[:-1],
            dtype=np.uint8,
        )

    def step(self, action):
        """Steps the environment by one step.

        Args:
            actions (Any): Agent's action.

        Returns:
            Tuple[ np.ndarray, float, bool, Dict[str, Any] ]:
                Observation, reward, done, info, of the agent.
        """
        obs, rewards, dones, infos = self.env.step(action)
        filtered = filter_obs(obs)
        return filtered, rewards, dones, infos

    def reset(self):
        """Resets the environment.

        Returns:
            np.ndarray: Agent's observation after reset.
        """
        obs = self.env.reset()
        filtered = filter_obs(obs)
        return filtered


def filter_obs(obs: format_obs.StdObs) -> np.ndarray:
    rgb = obs.rgb

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
    rgb = rgb.transpose(2, 0, 1)

    # Resize image to 64x64
    # resized = transposed[:, 96:160, 96:160]

    # rep = repainted.transpose(2, 0, 1)
    # plotter(rgb, 3, name="from obs")

    # sb3_util.plotter3d(rgb, rgb_gray=3, name="From Obs")

    return np.uint8(rgb)


class ObservationMLP(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        old_space = env.observation_space["rgb"]
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(old_space.shape[-1],) + old_space.shape[:-1],
            dtype=np.uint8,
        )

    def step(self, action):
        """Steps the environment by one step.

        Args:
            actions (Any): Agent's action.

        Returns:
            Tuple[ np.ndarray, float, bool, Dict[str, Any] ]:
                Observation, reward, done, info, of the agent.
        """
        obs, rewards, dones, infos = self.env.step(action)
        filtered = filter_obs(obs)
        return filtered, rewards, dones, infos

    def reset(self):
        """Resets the environment.

        Returns:
            np.ndarray: Agent's observation after reset.
        """
        obs = self.env.reset()
        filtered = filter_obs(obs)
        return filtered