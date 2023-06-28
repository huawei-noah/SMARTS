from typing import Callable, Tuple

import gymnasium as gym
import numpy as np

from smarts.core.controllers import ActionSpaceType


class FormatAction:
    def __init__(self, action_space_type: ActionSpaceType):
        if action_space_type == ActionSpaceType.Continuous:
            self._wrapper, self.action_space = _discrete_continuous()
        else:
            raise Exception(f"Unknown action space type {action_space_type}.")

    def format(self, action: int):
        """Adapts the action input to the wrapped environment.

        Note: Users should not directly call this method.
        """
        wrapped_act = self._wrapper(action)
        return wrapped_act


def _discrete_continuous() -> Tuple[Callable[[int], np.ndarray], gym.Space]:
    action_map = {
        # key: [throttle, brake, steering]
        0: [0, 0.5, 0],  # slow_down
        1: [0, 1, 0],  # slow_down
        2: [0.3, 0, 0],  # keep_direction
        3: [0.5, 0, 0],  # keep_direction
        4: [0.3, 0, -0.3],  # turn_left
        5: [0.3, 0, 0.3],  # turn_right
    }

    space = gym.spaces.Discrete(n=len(action_map))

    def wrapper(model_action: int) -> np.ndarray:
        throttle, brake, steering = action_map[model_action]
        return np.array([throttle, brake, steering], dtype=np.float32)

    return wrapper, space
