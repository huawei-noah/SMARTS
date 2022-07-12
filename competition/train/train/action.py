from typing import Any, Callable, Dict, Tuple

import gym
import numpy as np


class Action(gym.ActionWrapper):
    """Modifies the action space."""

    def __init__(self, env: gym.Env):
        """Sets identical action space, denoted by `space`, for all agents.

        Args:
            env (gym.Env): Gym env to be wrapped.
            space (str): Denotes the desired action space type.
        """
        super().__init__(env)
        space_map = {
            "Discrete": _discrete,
        }
        self._wrapper, action_space = space_map.get(space)()

        self.action_space = gym.spaces.Dict(
            {agent_id: action_space for agent_id in env.action_space.spaces.keys()}
        )

    def action(self, action):
        """Adapts the action input to the wrapped environment.

        `self.saved_obs` is retrieved from SaveObs wrapper. It contains previously
        saved observation parameters.

        Note: Users should not directly call this method.
        """
        wrapped_act = self._wrapper(action=action, saved_obs=self.saved_obs)

        return wrapped_act


def _discrete() -> Tuple[Callable[[Dict[str, int]], Dict[str, np.ndarray]], gym.Space]:
    space = gym.spaces.Discrete(n=4)

    time_delta = 0.1  # Time, in seconds, between steps.
    angle = 30 / 180 * np.pi  # Turning angle in radians
    speed = 40  # Speed in km/h
    dist = speed / 3600 * time_delta  # Distance travelled in time_delta seconds

    action_map = {
        # key: [magnitude, angle]
        0: [0, 0],  # slow_down
        1: [dist, 0],  # keep_direction
        2: [dist, angle],  # turn_left
        3: [dist, -angle],  # turn_right
    }

    def wrapper(
        action: Dict[str, int], saved_obs: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        wrapped_obs = {}
        for agent_id, agent_action in action.items():
            new_heading = saved_obs[agent_id].heading + action_map[agent_action][1]
            new_heading = (new_heading + np.pi) % (2 * np.pi) - np.pi
            magnitude = action_map[agent_action][0]
            # Note: On the map, angle is zero at positive y axis, and increases anti-clockwise.
            #       In np.exp(), angle is zero at positive x axis, and increases anti-clockwise.
            #       Hence, numpy_angle = map_angle + π/2
            new_pos = saved_obs[agent_id].pos[:2] + magnitude * np.exp(
                1j * (new_heading + np.pi / 2)
            )
            x_coord, y_coord = new_pos
            wrapped_obs.update(
                {
                    agent_id: np.array(
                        [x_coord, y_coord, new_heading, time_delta], dtype=np.float32
                    )
                }
            )
        return wrapped_obs

    return wrapper, space
