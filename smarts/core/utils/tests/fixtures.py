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
# FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
import pytest

from smarts.core.controllers import ActionSpaceType
from smarts.core.utils.dummy import dummy_observation


@pytest.fixture
def large_observation():
    return dummy_observation()


@pytest.fixture
def adapter_data():
    return [
        (ActionSpaceType.ActuatorDynamic, [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]),
        (ActionSpaceType.Continuous, [0.9, 0.8, 0.7], [0.9, 0.8, 0.7]),
        (ActionSpaceType.Lane, "keep_lane", "keep_lane"),
        (ActionSpaceType.LaneWithContinuousSpeed, [0, 20.2], [0, 20.2]),
        (
            ActionSpaceType.Trajectory,
            (
                [1, 2],
                [5, 6],
                [0.3, 3.14],
                [20.0, 21.0],
            ),
            (
                [166.23485529, 167.23485529],
                [2.2, 1.2],
                [-1.27079633, 1.56920367],
                [20.0, 21.0],
            ),
        ),
        (
            ActionSpaceType.TrajectoryWithTime,
            [
                [1, 2],
                [5, 6],
                [0.3, 3.14],
                [20.0, 21.0],
                [0.1, 0.2],
            ],
            [
                [166.23485529, 167.23485529],
                [2.2, 1.2],
                [-1.27079633, 1.56920367],
                [20.0, 21.0],
                [0.1, 0.2],
            ],
        ),
        (
            ActionSpaceType.MPC,
            [
                [1, 2],
                [5, 6],
                [0.3, 3.14],
                [20.0, 21.0],
            ],
            [
                [166.23485529, 167.23485529],
                [2.2, 1.2],
                [-1.27079633, 1.56920367],
                [20.0, 21.0],
            ],
        ),
        (
            ActionSpaceType.TargetPose,
            (2, 4, -2.9, 20),
            (165.23485529, 1.2, 1.81238898, 20.0),
        ),
        (ActionSpaceType.Direct, (2, 2), (2, 2)),
    ]
