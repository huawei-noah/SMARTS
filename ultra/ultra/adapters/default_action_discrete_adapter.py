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
        super(DiscreteStrings, self).__init__((), np.str)

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


gym_space: gym.Space = DiscreteStrings(
    ("keep_lane", "slow_down", "change_lane_left", "change_lane_right")
)
required_interface = {"action": ActionSpaceType.Lane}


def adapt(action):
    return action
