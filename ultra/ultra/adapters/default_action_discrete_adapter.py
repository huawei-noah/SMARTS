# MIT License
#
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
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
from typing import Sequence

import gym
import numpy as np

from smarts.core.controllers import ActionSpaceType


class DiscreteStrings(gym.Space):
    def __init__(self, strings: Sequence[str]):
        assert len(strings) > 0
        self._strings = strings
        super(DiscreteStrings, self).__init__((), str)

    def sample(self) -> str:
        index = self.np_random.randint(len(self._strings))
        return self._strings[index]

    def contains(self, x: str) -> bool:
        return x in self._strings

    def __repr__(self) -> str:
        strings_string = ", ".join(self._strings)
        return f"DiscreteStrings(({strings_string}))"

    def __eq__(self, other) -> bool:
        return isinstance(other, DiscreteStrings) and self._strings == other._strings


# The space of the adapted action.
gym_space: gym.Space = DiscreteStrings(
    ("keep_lane", "slow_down", "change_lane_left", "change_lane_right")
)
# This adapter requires SMARTS to ensure that the agent is provided a "lane" controller,
# that is, a controller that allows for actions in the form of one of four valid strings
# that describe the action of the agent. The four valid strings are "keep-lane",
# "slow_down", "change_lane_left", and "change_lane_right".
required_interface = {"action": ActionSpaceType.Lane}


def adapt(action: str) -> str:
    """Adapts a given action into an action that SMARTS can understand for a lane
    controller. This adapter expects that the action is already a valid lane controller
    action.

    Args:
        action (str): The action to adapt. The action should be one of four applicable
            strings, either "slow_down", "keep_lane", "change_lane_left", or
            "change_lane_right".

    Returns:
        str: The same action that was passed in.
    """
    return action
