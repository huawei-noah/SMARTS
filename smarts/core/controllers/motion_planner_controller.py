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
# FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
from typing import Optional, Tuple

import numpy as np

from smarts.core.bezier_motion_planner import BezierMotionPlanner
from smarts.core.chassis import BoxChassis
from smarts.core.coordinates import Heading, Pose


class MotionPlannerControllerState:
    """Controller state"""

    def __init__(self):
        self._motion_planner: BezierMotionPlanner = BezierMotionPlanner()

    def get_next_state(
        self,
        cur_pose: Pose,
        cur_speed: float,
        dt: float,
        target_pose_at_t: Optional[np.ndarray],
    ) -> Tuple[Pose, float]:
        """Computes a cubic bezier curve to the target_pose_at_t."""
        cur_state = np.array(
            [*cur_pose.position[:2], float(cur_pose.heading), cur_speed]
        ).astype(float)
        if target_pose_at_t is None:
            # if agent failed to produce a target pose, just use the previous pose
            target_pose_at_t = cur_state
        new_traj_pt = self._motion_planner.trajectory(
            cur_state, target_pose_at_t, 1, dt
        ).reshape(4)
        new_pose = Pose.from_center(new_traj_pt[:2], Heading(new_traj_pt[2]))
        return new_pose, new_traj_pt[3]


class MotionPlannerController:
    """A controller that shapes agent vehicle motion to follow a motion plan."""

    @classmethod
    def perform_action(
        cls,
        controller_state: MotionPlannerControllerState,
        dt: float,
        vehicle,
        action: Optional[np.ndarray],
    ):
        """Performs an action adapting to the underlying chassis.

        Args:
            controller_state (MotionPlannerControllerState): The previous controller state from this controller.
            dt (float): Amount of time that has passed since the last action.
            vehicle (Vehicle): Vehicle to control.
            action: Pose denoted by [x_coordinate, y_coordinate, heading, t_s], at t_s seconds into future.
        """
        assert isinstance(vehicle.chassis, BoxChassis)
        assert len(action) >= 4, f"{action}"
        pose, speed = controller_state.get_next_state(
            vehicle.pose, vehicle.speed, dt, action
        )
        vehicle.control(pose, speed, dt)
