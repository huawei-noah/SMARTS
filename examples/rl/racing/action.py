from typing import Callable, Tuple

import gym
import numpy as np


class Action(gym.ActionWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._wrapper, action_space = continuous()
        self.action_space = gym.spaces.Dict(
            {agent_id: action_space for agent_id in env.agent_specs.keys()}
        )

    def action(self, act):
        wrapped_act = {
            agent_id: self._wrapper(agent_act) for agent_id, agent_act in act.items()
        }
        return wrapped_act


def continuous() -> Tuple[Callable[[np.ndarray], np.ndarray], gym.Space]:
    space = gym.spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

    def wrapper(model_action):
        throttle, brake, steering = model_action
        throttle = (throttle + 1) / 2
        brake = (brake + 1) / 2
        return np.array([throttle, brake, steering], dtype=np.float32)

    return wrapper, space
