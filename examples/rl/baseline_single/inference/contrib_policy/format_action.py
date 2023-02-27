from typing import Callable, Tuple

import gymnasium as gym
import numpy as np


class FormatAction:
    def __init__(self):
        self._wrapper, self.action_space = _discrete()

    def format(self, action):
        """Adapts the action input to the wrapped environment.

        Note: Users should not directly call this method.
        """
        wrapped_act = self._wrapper(action)
        return wrapped_act


def _discrete() -> Tuple[Callable[[int], np.ndarray], gym.Space]:
    space = gym.spaces.Discrete(n=4)

    action_map = {
        # key: [throttle, brake, steering]
        0: [0.3, 0, 0],  # keep_direction
        1: [0, 1, 0],  # slow_down
        2: [0.3, 0, -0.5],  # turn_left
        3: [0.3, 0, 0.5],  # turn_right
    }

    def wrapper(model_action: int) -> np.ndarray:
        throttle, brake, steering = action_map[model_action]
        return np.array([throttle, brake, steering], dtype=np.float32)

    return wrapper, space
