from typing import Dict

import gymnasium as gym
import numpy as np


class MakeDict:
    """Converts gym.spaces.Box to gym.spaces.Dict."""

    def __init__(self, input_space: gym.Space):
        self.observation_space = gym.spaces.Dict({"rgb": input_space})

    def make(self, obs: np.ndarray) -> Dict[str, np.ndarray]:
        """converts input into a dict.

        Args:
            obs (np.ndarray): Numpy array input.

        Returns:
            dict[str, np.ndarray]: A dictionary with key (string) corresponding
                to input obs (np.ndarray).
        """
        return {"rgb": obs}
