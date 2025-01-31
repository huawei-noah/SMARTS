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
from __future__ import annotations


import logging
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING

import gymnasium as gym
import numpy as np

from smarts.core.coordinates import Heading

from .limit_relative_target_pose import LimitRelativeTargetPose

logger = logging.getLogger(__file__)
logger.setLevel(logging.WARNING)

if TYPE_CHECKING:
    from smarts.core.smarts import SMARTS


class LimitTargetPose(LimitRelativeTargetPose):
    """Limits the delta-x and delta-y in the RelativeTargetPose action space."""

    def __init__(self, env: gym.Env):
        """
        Args:
            env (gym.Env): Environment to be wrapped.
        """
        super().__init__(env)

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

        engine: SMARTS = self.env.unwrapped.smarts
        agent_vehicles = {a_id: engine.vehicle_index.vehicles_by_owner_id(a_id) for a_id in action}
        limited_actions: Dict[str, np.ndarray] = {}
        for agent_name, agent_action in action.items():
            # TODO: support timesteps other than 0.1
            # time_ratio = engine.fixed_timestep_sec
            assert agent_action[3] == 0.1
            current_vehicles = agent_vehicles.get(agent_name)
            if len(current_vehicles) != 1:
                continue
            pose = current_vehicles[0].pose
            limited_relative_action = self._limit(
                name=agent_name,
                action=np.array([
                    agent_action[0] - pose.position[0],
                    agent_action[1] - pose.position[1],
                    float(Heading(agent_action[2]).relative_to(pose.heading))
                ]),
            )

            limited_actions[agent_name] = np.array([
                pose.position[0] + limited_relative_action[0],
                pose.position[1] + limited_relative_action[1],
                float(Heading(pose.heading + limited_relative_action[2])),
                agent_action[3],
            ])

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

        _limited_relative_action = super()._limit(name, action)

        return _limited_relative_action
