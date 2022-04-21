from typing import Callable, Tuple

import gym
import numpy as np


class Action(gym.ActionWrapper):
    def __init__(self, env: gym.Env, space:str):
        super().__init__(env)
        if space == "Continuous":
            self._wrapper, self.action_space = _continuous()
        elif space == "Lane":
            self._wrapper, self.action_space = _lane()
        elif space == "Discrete":
            self._wrapper, self.action_space = _discrete()            
        else:
            raise Exception(f"Unknown action space {space}.")

    def action(self, action):
        """Adapts the action input to the wrapped environment.

        Note: Users should not directly call this method.
        """
        wrapped_act = self._wrapper(action)
        return wrapped_act


def _continuous() -> Tuple[Callable[[np.ndarray], np.ndarray], gym.Space]:
    space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

    def wrapper(model_action: np.ndarray) -> np.ndarray:
        throttle, brake, steering = model_action
        throttle = (throttle + 1) / 2
        brake = (brake + 1) / 2
        return np.array([throttle, brake, steering], dtype=np.float32)

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

def _discrete() -> Tuple[Callable[[int], np.ndarray], gym.Space]:
    space = gym.spaces.Discrete(n=4)

    action_map = {
        # key: [throttle, brake, steering]
        0: [0.5, 0, 0], # keep_direction
        1: [0, 1, 0], # slow_down
        2: [0.5, 0, -0.8], # turn_left
        3: [0.5, 0, 0.8], # turn_right
    }

    def wrapper(model_action: int) -> np.ndarray:
        throttle, brake, steering = action_map[model_action]
        return np.array([throttle, brake, steering], dtype=np.float32)

    return wrapper, space