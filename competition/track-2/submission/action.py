from typing import Callable, Dict, Tuple

import gym
import numpy as np


class Action(gym.ActionWrapper):
    """Modifies the action space."""

    def __init__(self, env: gym.Env, space: str):
        """Sets identical action space, denoted by `space`, for all agents.
        Args:
            env (gym.Env): Gym env to be wrapped.
            space (str): Denotes the desired action space type.
        """
        super().__init__(env)
        space_map = {
            "Discrete": _discrete(),
        }
        # breakpoint()
        self._wrapper, action_space = space_map.get(space)
        #action_space = space_map.get(space)[1]

        self.action_space = gym.spaces.Dict(
            {agent_id: action_space for agent_id in env.action_space.spaces.keys()}
        )

    def action(self, action):
        """Adapts the action input to the wrapped environment.
        Note: Users should not directly call this method.
        """
        return action


def _discrete() -> Tuple[Callable[[Dict[str, int]], Dict[str, np.ndarray]], gym.Space]:
    space = gym.spaces.Discrete(n=4)

    action_map = {
        # key: [throttle, brake, steering]
        0: np.array([0.3, 0, 0], dtype=np.float32),  # keep_direction
        1: np.array([0, 1, 0], dtype=np.float32),  # slow_down
        2: np.array([0.3, 0, -0.5], dtype=np.float32),  # turn_left
        3: np.array([0.3, 0, 0.5], dtype=np.float32),  # turn_right
    }

    def wrapper(action: Dict[str, int]) -> Dict[str, np.ndarray]:
        final = {}
        for agent_id, agent_action in action.items():
            final.update({agent_id, action_map[agent_action]})
        return final
        # return {
        #     agent_id: action_map[agent_action] for agent_id, agent_action in action.items()
        # }

    return wrapper, space
