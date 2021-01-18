# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
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

from smarts.core.coordinates import Heading, Pose
from smarts.core.chassis import AckermannChassis, BoxChassis

from smarts.core.controllers.trajectory_tracking_controller import (
    TrajectoryTrackingControllerState,
    TrajectoryTrackingController,
)

TIME_INDEX = 0
X_INDEX = 1
Y_INDEX = 2
THETA_INDEX = 3
VEL_INDEX = 4


class TrajectoryInterpolationController:
    @staticmethod
    def ndarray_has_nan(ndarray: np.ndarray):
        np.isnan(np.sum(ndarray))

    @staticmethod
    def interpolate(ms0: np.ndarray, ms1: np.ndarray, time: float) -> np.ndarray:
        """Linear Interpolate between two vehicle motion state

        Returns:
            np.ndarray: New vehicle state between vehicle motion state ms0 and ms1
        """

        start_time = ms0[TIME_INDEX]
        end_time = ms1[TIME_INDEX]
        ratio = math.fabs((time - start_time) / (end_time - start_time))
        assert end_time >= start_time and time >= start_time

        np_motion_state = (1 - ratio) * ms0 + ratio * ms1
        return np_motion_state

    @staticmethod
    def locate_motion_state(trajectory, time) -> np.ndarray:

        if math.isinf(trajectory[TIME_INDEX][0]):
            return trajectory[:, 0], trajectory[:, 0]

        end_index = 0
        for i, t in enumerate(trajectory[TIME_INDEX]):
            if t > time:
                end_index = i
                break

        if end_index <= 0:
            raise RuntimeError(
                "Next step time is %f, but cant be located at input traj" % time
            )
        return trajectory[:, end_index - 1], trajectory[:, end_index]

    @classmethod
    def perform_trajectory_interpolation(
        cls, sim, agent_id, vehicle, trajectory: np.ndarray, controller_state,
    ):
        """Move vehicle with trajectory interpolation

        If you want vehicle stop at a specific pose,
        trajectory[TIME_INDEX][0] should be set as infinite

        Args:
            sim : reference of smarts instance
            agent_id : agent who use this controller
            vehicle : vehicle to be controlled
            trajectory (np.ndarray): trajectory with time
            controller_state : inner state of controller

        Raises:
            ValueError: Error when input trajectory has nan field.
            RuntimeError: Error when input trajectory is empty
        """
        assert isinstance(vehicle.chassis, BoxChassis)
        if len(trajectory[TIME_INDEX]) <= 0:
            raise RuntimeError("Input trajectory is empty!")
        if cls.ndarray_has_nan(trajectory):
            raise ValueError("Has nan in trajectory")

        ms0, ms1 = cls.locate_motion_state(trajectory, sim._elapsed_sim_time)

        speed = 0.0
        pose = []
        if math.isinf(ms0[TIME_INDEX]) or math.isinf(ms1[TIME_INDEX]):
            center_position = ms0[X_INDEX : Y_INDEX + 1]
            center_heading = Heading(ms0[THETA_INDEX])
            pose = Pose.from_center(center_position, center_heading)
            speed = 0.0
        else:
            ms = cls.interpolate(ms0, ms1, sim._elapsed_sim_time)

            center_position = ms[X_INDEX : Y_INDEX + 1]
            center_heading = Heading(ms[THETA_INDEX])
            pose = Pose.from_center(center_position, center_heading)
            speed = ms[VEL_INDEX]

        vehicle.set_pose(pose)
        vehicle.set_speed(speed)
