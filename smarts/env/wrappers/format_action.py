# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from typing import Callable, Dict, Tuple

import gym
import numpy as np

from smarts.core.controllers import ActionSpaceType


class FormatAction(gym.ActionWrapper):
    """Sets gym-compliant action space for SMARTS environment.

    Note:

        (a) Only ``ActionSpaceType.Continuous``, ``ActionSpaceType.Lane``, and
            ``ActionSpaceType.TargetPose`` are supported by this wrapper now.

        (b) All agents should have the same action space.
    """

    def __init__(self, env: gym.Env, space: ActionSpaceType):
        """Sets identical action space, denoted by ``space``, for all agents.

        :param env: Gym env to be wrapped.
        :type env: class: gym.Env
        :param space: Denotes the desired action space type from
                ``smarts.core.controllers.ActionSpaceType``.
        :type space: str
        """
        super().__init__(env)
        space_map = {
            "Continuous": _continuous,
            "Lane": _lane,
            "TargetPose": _target_pose,
        }
        self._wrapper, action_space = space_map.get(space.name)()

        self.action_space = gym.spaces.Dict(
            {agent_id: action_space for agent_id in self.agent_interfaces.keys()}
        )

    def action(self, action):
        """Adapts the action input to the wrapped environment.

        Note: Users should not directly call this method.
        """
        wrapped_act = self._wrapper(action)
        return wrapped_act


def _continuous() -> Tuple[
    Callable[[Dict[str, np.ndarray]], Dict[str, np.ndarray]], gym.Space
]:
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


def _target_pose() -> Tuple[
    Callable[[Dict[str, np.ndarray]], Dict[str, np.ndarray]], gym.Space
]:
    space = gym.spaces.Box(
        low=np.array([-1e10, -1e10, -np.pi, 0]),
        high=np.array([1e10, 1e10, np.pi, 1e10]),
        dtype=np.float32,
    )

    def wrapper(action: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return {k: v.astype(np.float32) for k, v in action.items()}

    return wrapper, space
