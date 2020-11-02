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
from enum import Enum
from functools import partial

from numpy.linalg import matrix_power
import numpy as np

from scipy import signal

from smarts.core.chassis import AckermannChassis
from smarts.core.utils.math import (
    lerp,
    radians_to_vec,
    signed_dist_to_line,
    min_angles_difference_signed,
    low_pass_filter,
)
from smarts.core.controllers.trajectory_tracking_controller import (
    TrajectoryTrackingControllerState,
    TrajectoryTrackingController,
)
from smarts.core.controllers.lane_following_controller import (
    LaneFollowingControllerState,
    LaneFollowingController,
)
from smarts.core.controllers.actuator_dynamic_controller import (
    ActuatorDynamicControllerState,
    ActuatorDynamicController,
)

METER_PER_SECOND_TO_KM_PER_HR = 3.6


class ActionSpaceType(Enum):
    Continuous = 0
    Lane = 1
    ActuatorDynamic = 2
    LaneWithContinuousSpeed = 3
    TargetPose = 4
    Trajectory = 5
    MultiTargetPose = 6  # for boid control
    MPC = 7


class Controllers:
    @staticmethod
    def perform_action(
        sim, agent_id, vehicle, action, controller_state, sensor_state, action_space
    ):
        if action is None:
            return

        if action_space == ActionSpaceType.Continuous:
            vehicle.control(
                throttle=np.clip(action[0], 0.0, 1.0),
                brake=np.clip(action[1], 0.0, 1.0),
                steering=np.clip(action[2], -1, 1),
            )
        elif action_space == ActionSpaceType.ActuatorDynamic:
            ActuatorDynamicController.perform_action(
                vehicle, action, controller_state, dt_sec=sim.timestep_sec
            )
        elif action_space == ActionSpaceType.Trajectory:
            TrajectoryTrackingController.perform_trajectory_tracking_PD(
                action, vehicle, controller_state, dt_sec=sim.timestep_sec,
            )
        elif action_space == ActionSpaceType.MPC:
            TrajectoryTrackingController.perform_trajectory_tracking_MPC(
                action, vehicle, controller_state, sim.timestep_sec
            )
        elif action_space == ActionSpaceType.LaneWithContinuousSpeed:
            LaneFollowingController.perform_lane_following(
                sim,
                agent_id,
                vehicle,
                controller_state,
                sensor_state,
                action[0],
                action[1],
            )
        elif action_space == ActionSpaceType.Lane:
            perform_lane_following = partial(
                LaneFollowingController.perform_lane_following,
                sim=sim,
                agent_id=agent_id,
                vehicle=vehicle,
                controller_state=controller_state,
                sensor_state=sensor_state,
            )

            # 12.5 m/s (45 km/h) is used as the nominal speed for lane change.
            # For keep_lane, the nominal speed is set to 15 m/s (54 km/h).
            if action == "keep_lane":
                perform_lane_following(target_speed=15, lane_change=0)
            elif action == "slow_down":
                perform_lane_following(target_speed=0, lane_change=0)
            elif action == "change_lane_left":
                perform_lane_following(target_speed=12.5, lane_change=1)
            elif action == "change_lane_right":
                perform_lane_following(target_speed=12.5, lane_change=-1)
        else:
            raise ValueError(
                f"perform_action(action_space={action_space}, ...) has failed "
                "inside controller"
            )


class ControllerState:
    @staticmethod
    def from_action_space(action_space, vehicle_position, sim):
        if action_space == ActionSpaceType.Lane:
            # TAI: we should probably be fetching these waypoint through the mission planner
            target_lane_id = sim.waypoints.closest_waypoint(vehicle_position).lane_id
            return LaneFollowingControllerState(target_lane_id)

        if action_space == ActionSpaceType.LaneWithContinuousSpeed:
            # TAI: we should probably be fetching these waypoint through the mission planner
            target_lane_id = sim.waypoints.closest_waypoint(vehicle_position).lane_id
            return LaneFollowingControllerState(target_lane_id)

        if action_space == ActionSpaceType.ActuatorDynamic:
            return ActuatorDynamicControllerState()

        if action_space == ActionSpaceType.Trajectory:
            return TrajectoryTrackingControllerState()

        if action_space == ActionSpaceType.MPC:
            return TrajectoryTrackingControllerState()

        # Other action spaces do not need a controller state object
        return None
