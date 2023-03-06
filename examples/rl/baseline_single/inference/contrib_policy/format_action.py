from typing import Callable, Tuple

import gym
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
    action_map = {
        # key: [throttle, brake, steering]
        0: [0, 1, 0],  # slow_down
        1: [0.3, 0, 0],  # accelerate_forward
        2: [0.3, 0, -0.3],  # turn_left
        3: [0.3, 0, 0.3],  # turn_right
    }

    space = gym.spaces.Discrete(n=len(action_map))

    def wrapper(model_action: int) -> np.ndarray:
        throttle, brake, steering = action_map[model_action]
        return np.array([throttle, brake, steering], dtype=np.float32)

    return wrapper, space
