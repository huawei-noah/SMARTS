from typing import Callable, Tuple

import gym
import numpy as np


class FormatAction(gym.ActionWrapper):
    def __init__(self, env: gym.Env, space: str):
        super().__init__(env)
        space_map = {"Continuous": _continuous, "Lane": _lane}
        self._wrapper, self.action_space = space_map.get(space)()

    def action(self, action):
        """Adapts the action input to the wrapped environment.

        Note: Users should not directly call this method.
        """
        wrapped_act = self._wrapper(action)
        return wrapped_act


def _continuous() -> Tuple[Callable[[np.ndarray], np.ndarray], gym.Space]:
    space = gym.spaces.Box(
        low=np.array([0.0, 0.0, -1.0]), high=np.array([1.0, 1.0, 1.0]), dtype=np.float32
    )

    def wrapper(model_action: np.ndarray) -> np.ndarray:
        return model_action.astype(np.float32)

    return wrapper, space


def _lane() -> Tuple[Callable[[int], str], gym.Space]:
    space = gym.spaces.Discrete(n=4)

    action_map = {
        0: "keep_lane",
        1: "slow_down",
        2: "change_lane_left",
        3: "change_lane_right",
    }

    def wrapper(model_action: int) -> str:
        return action_map[model_action]

    return wrapper, space
