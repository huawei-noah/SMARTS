from typing import Callable, Tuple

import gym
import numpy as np


class Action(gym.ActionWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._wrapper, self.action_space = _continuous()

    def action(self, action):
        """Adapts the action input to the wrapped environment.

        Note: Users should not directly call this method.
        """
        wrapped_act = self._wrapper(action)
        return wrapped_act


def _continuous() -> Tuple[Callable[[np.array], np.array], gym.Space]:
    space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

    def wrapper(model_action):
        throttle, brake, steering = model_action
        throttle = (throttle + 1) / 2
        brake = (brake + 1) / 2
        return np.array([throttle, brake, steering], dtype=np.float32)

    return wrapper, space
