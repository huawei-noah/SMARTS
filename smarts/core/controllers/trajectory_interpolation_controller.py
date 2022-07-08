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
import math
from enum import IntEnum
from typing import Optional, Tuple

import numpy as np

from smarts.core.chassis import BoxChassis
from smarts.core.coordinates import Heading, Pose


class TrajectoryField(IntEnum):
    """Used to access fields from trajectory data."""

    TIME_INDEX = 0
    X_INDEX = 1
    Y_INDEX = 2
    THETA_INDEX = 3
    VEL_INDEX = 4


class TrajectoryInterpolationController:
    """A controller used to perform trajectory interpolation."""

    @classmethod
    def assert_is_legal_trajectory(cls, trajectory: np.ndarray):
        """Test if the trajectory is correctly formed."""
        assert (
            len(trajectory[TrajectoryField.TIME_INDEX]) >= 2
        ), f"length of trajectory is {len(trajectory[TrajectoryField.TIME_INDEX])}"

        assert np.isfinite(
            trajectory
        ).all(), "trajectory has nan, positive inf or negative inf"

        assert (
            np.diff(trajectory[TrajectoryField.TIME_INDEX]) > 0
        ).all(), "trajectory times are not strictly increasing!"

    @classmethod
    def _interpolate(cls, ms0: np.ndarray, ms1: np.ndarray, time: float) -> np.ndarray:
        """Linearly interpolate between two vehicle motion states.

        Returns:
            np.ndarray: New vehicle state between vehicle motion state ms0 and ms1
        """
        start_time = ms0[TrajectoryField.TIME_INDEX]
        end_time = ms1[TrajectoryField.TIME_INDEX]
        assert (
            end_time >= start_time and time >= start_time
        ), f"{start_time} <= {time} <= {end_time} ?"
        ratio = math.fabs((time - start_time) / (end_time - start_time))
        left_over = 1.0 - ratio

        np_motion_state = left_over * ms0 + ratio * ms1
        CS = left_over * math.cos(ms0[TrajectoryField.THETA_INDEX]) + ratio * math.cos(
            ms1[TrajectoryField.THETA_INDEX]
        )
        SN = left_over * math.sin(ms0[TrajectoryField.THETA_INDEX]) + ratio * math.sin(
            ms1[TrajectoryField.THETA_INDEX]
        )
        np_motion_state[TrajectoryField.THETA_INDEX] = math.atan2(SN, CS)
        return np_motion_state

    @classmethod
    def _locate_motion_state(
        cls, trajectory: np.ndarray, time: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Find the first pair of motion states within the given trajectory where the second is later than given time."""
        for i, t in enumerate(trajectory[TrajectoryField.TIME_INDEX]):
            if t > time:
                return trajectory[:, i - 1], trajectory[:, i]
        assert (
            False
        ), f"could not locate points within the trajectory that span {time} secs from now"
        return trajectory[:, 0], trajectory[:, 1]

    @classmethod
    def perform_action(cls, dt: float, vehicle, trajectory: np.ndarray):
        """Move vehicle by trajectory interpolation.

        If you want vehicle stop at a specific pose,
        trajectory[TrajectoryField.TIME_INDEX][0] should be set as numpy.inf.

        Args:
            dt (float): the amount of time that is expected to pass between actions
            vehicle (Vehicle) : vehicle to be controlled
            trajectory (np.ndarray) : trajectory with 5 dimensions - TIME, X, Y, THETA and VEL
        """
        assert isinstance(vehicle.chassis, BoxChassis)
        cls.assert_is_legal_trajectory(trajectory)

        ms0, ms1 = cls._locate_motion_state(trajectory, dt)

        if math.isinf(ms0[TrajectoryField.TIME_INDEX]) or math.isinf(
            ms1[TrajectoryField.TIME_INDEX]
        ):
            ms = ms0
            speed = 0.0
        else:
            ms = cls._interpolate(ms0, ms1, dt)
            speed = ms[TrajectoryField.VEL_INDEX]

        center_position = ms[TrajectoryField.X_INDEX : TrajectoryField.Y_INDEX + 1]
        center_heading = Heading(ms[TrajectoryField.THETA_INDEX])
        pose = Pose.from_center(center_position, center_heading)

        vehicle.control(pose, speed, dt)
