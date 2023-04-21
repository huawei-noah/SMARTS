# Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import logging
from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np

logger = logging.getLogger(__file__)
logger.setLevel(logging.WARNING)


class LimitRelativeTargetPose(gym.Wrapper):
    """Limits the delta-x and delta-y in the RelativeTargetPose action space."""

    def __init__(self, env: gym.Env):
        """
        Args:
            env (gym.Env): Environment to be wrapped.
        """
        super().__init__(env)
        self._time_delta = 0.1
        self._speed_max = 22.22  # Units: m/s. Equivalent to 80 km/h.
        self._dist_max = self._speed_max * self._time_delta

    def step(
        self, action: Dict[str, np.ndarray]
    ) -> Tuple[
        Dict[str, Any],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Dict[str, Any]],
    ]:
        """Steps the environment.

        Args:
            action (Dict[str, np.ndarray]): Action for each agent.

        Returns:
            Tuple[ Dict[str, Any], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Dict[str, Any]] ]:
                Observation, reward, terminated, truncated, and info, for each agent is returned.
        """

        limited_actions: Dict[str, np.ndarray] = {}
        for agent_name, agent_action in action.items():
            limited_actions[agent_name] = self._limit(
                name=agent_name,
                action=agent_action,
            )

        out = self.env.step(limited_actions)
        return out

    def _limit(
        self,
        name: str,
        action: np.ndarray,
    ) -> np.ndarray:
        """Limit Euclidean distance travelled in RelativeTargetPose action space.

        Args:
            name (str): Agent's name.
            action (np.ndarray): Agent's action.

        Returns:
            np.ndarray: Agent's RelativeTargetPose action with constrained delta-x and delta-y coordinates.
        """

        limited_action = np.array([action[0], action[1], action[2]], dtype=np.float32)
        vector = action[:2]
        dist = np.linalg.norm(vector)
        if dist > self._dist_max:
            unit_vector = vector / dist
            limited_action[0], limited_action[1] = self._dist_max * unit_vector
            logger.warning(
                "Action out of bounds. `%s`: Allowed max speed=%sm/s, but got speed=%sm/s. "
                "Action changed from %s to %s.",
                name,
                self._speed_max,
                dist / self._time_delta,
                action,
                limited_action,
            )

        return limited_action
