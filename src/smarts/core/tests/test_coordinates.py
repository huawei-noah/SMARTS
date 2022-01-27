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
import math

import numpy as np
import pytest

from smarts.core.coordinates import Heading, Pose


@pytest.fixture
def original_pose():
    return Pose(position=np.array([0, 1, 0]), orientation=np.array([0, 0, 0, 1]))


def test_conversion_sumo(original_pose):
    pose_info = [
        ([2, 0], 90, 4),
        ([-1, 0], 270, 2),
        ([5, 5], 45, math.sqrt(50) * 2),
        ([0, -1.5], 180, 3),
    ]

    # spin around a center point
    for offset, angle, length in pose_info:
        front_bumper_pos = np.array(
            [
                original_pose.position[0] + offset[0],
                original_pose.position[1] + offset[1],
            ]
        )
        p_from_bumper = Pose.from_front_bumper(
            front_bumper_pos,
            Heading.from_sumo(angle),
            length,
        )

        assert np.isclose(
            p_from_bumper.position, original_pose.position, atol=2e-06
        ).all()

        _pose_position, _pose_heading = p_from_bumper.as_sumo(length, Heading(0))

        assert math.isclose(
            angle,
            _pose_heading,
            rel_tol=1e-06,
        )
        assert np.isclose(
            np.append(front_bumper_pos, 0),
            _pose_position,
            rtol=1e-06,
        ).all()


def test_conversion_panda(original_pose):
    pose_info = [
        (
            [0, 1, 0],
            0,
        ),
        (
            [0, 1, 0],
            180,
        ),
        (
            [0, 1, 0],
            180,
        ),
        (
            [0, 1],
            90,
        ),
    ]

    for position, heading in pose_info:
        heading = Heading.from_panda3d(heading)
        p_from_center = Pose.from_center(base_position=position, heading=heading)

        assert len(p_from_center.position) == 3
        assert len(p_from_center.orientation) == 4

        a_pos, a_ori = p_from_center.position, p_from_center.heading
        if len(position) < 3:
            position = [*position, 0]
        assert np.isclose(position, a_pos, atol=2e-07).all()
        assert math.isclose(heading, a_ori, abs_tol=2e-07)


def test_conversion_bullet(original_pose):
    pose_info = [
        (
            [2, 0, 0],
            0,
            [-2, 0, 0],
        ),
        ([-1, 0, 0], math.pi, [-1, 0, 0]),
        (
            [0, -1.5, 0],
            math.pi * 0.5,
            [-1.5, 0, 0],
        ),
    ]

    for offset_to_center, heading, position_offset in pose_info:
        heading = Heading.from_bullet(heading)
        offset_to_center = np.array(offset_to_center)
        base_position = np.sum([position_offset, original_pose.position], axis=0)
        p_from_explicit_offset = Pose.from_explicit_offset(
            offset_from_centre=offset_to_center,
            base_position=base_position,
            heading=heading,
            local_heading=Heading(0),
        )
        assert np.isclose(
            p_from_explicit_offset.position, original_pose.position, atol=2e-07
        ).all()
        assert math.isclose(
            p_from_explicit_offset.position[0], original_pose.position[0], abs_tol=2e-07
        )


def test_coordinates_heading():

    assert Heading() == 0

    assert Heading(-3.14) == -3.14

    assert Heading(-math.pi - 1) == math.pi - 1

    assert Heading(math.pi + 1) == -math.pi + 1

    assert math.isclose(
        Heading(math.pi / 4).relative_to(Heading(math.pi)), Heading(-2.356194490192345)
    )
    assert math.isclose(
        Heading(0).relative_to(Heading(math.pi + 1)), Heading(math.pi - 1)
    )
    assert math.isclose(
        Heading(math.pi + 1).relative_to(Heading(0)), Heading(-math.pi + 1)
    )
    assert math.isclose(
        Heading(math.pi + 1).relative_to(Heading(-math.pi - 1)), Heading(2)
    )

    assert math.isclose(
        Heading(2 * math.pi).relative_to(Heading(-2 * math.pi)), Heading(0)
    )
    assert math.isclose(
        Heading(2 * math.pi).relative_to(Heading(-2 * math.pi - 1)), Heading(1)
    )
    assert math.isclose(
        Heading(2 * math.pi).relative_to(Heading(4 * math.pi)), Heading(0)
    )
    assert math.isclose(
        Heading(-2 * math.pi).relative_to(Heading(-4 * math.pi)), Heading(0)
    )
