from typing import Callable, Dict, Tuple

import gym
import numpy as np

from smarts.core.controllers import ActionSpaceType


class FormatAction(gym.ActionWrapper):
    """Sets gym-compliant action space for SMARTS environment. 
        
    Note:
        (a) Only "ActionSpaceType.Continuous" and "ActionSpaceType.Lane"
            are supported by this wrapper now.
        (c) All agents should have the same action space.
        (b) Action adapters should not be used inside the `step` method of the 
            base environment.
    """
    
    def __init__(self, env: gym.Env, space: ActionSpaceType):
        """Sets identical action space, denoted by `space`, for all agents.
        
        Args:
            env (gym.Env): Gym env to be wrapped.
            space (str): Denotes the desired action space type from
                `smarts.core.controllers.ActionSpaceType`.
        """
        super().__init__(env)
        space_map = {"Continuous": _continuous, "Lane": _lane}
        self._wrapper, action_space = space_map.get(space.name)()

        self.action_space = gym.spaces.Dict(
            {agent_id: action_space for agent_id in self.agent_specs.keys()}
        )

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

    def wrapper(action: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return {k: v.astype(np.float32) for k, v in action.items()}

    return wrapper, space


def _lane() -> Tuple[Callable[[Dict[str, int]], Dict[str, str]], gym.Space]:
    space = gym.spaces.Discrete(n=4)

    action_map = {
        0: "keep_lane",
        1: "slow_down",
        2: "change_lane_left",
        3: "change_lane_right",
    }

    def wrapper(action: Dict[str, int]) -> Dict[str, str]:
        return {k: action_map[v] for k, v in action.items()}

    return wrapper, space
