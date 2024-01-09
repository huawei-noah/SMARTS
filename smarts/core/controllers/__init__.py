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
import enum
from functools import partial
from typing import Dict, Literal, Sequence, Tuple, Union

import numpy as np

from smarts.core.controllers.actuator_dynamic_controller import (
    ActuatorDynamicController,
    ActuatorDynamicControllerState,
)
from smarts.core.controllers.direct_controller import DirectController
from smarts.core.controllers.lane_following_controller import (
    LaneFollowingController,
    LaneFollowingControllerState,
)
from smarts.core.controllers.motion_planner_controller import (
    MotionPlannerController,
    MotionPlannerControllerState,
)
from smarts.core.controllers.trajectory_interpolation_controller import (
    TrajectoryInterpolationController,
)
from smarts.core.controllers.trajectory_tracking_controller import (
    TrajectoryTrackingController,
    TrajectoryTrackingControllerState,
)

from .action_space_type import ActionSpaceType

METER_PER_SECOND_TO_KM_PER_HR = 3.6


class LaneAction(enum.Enum):
    """The action for lane space actions."""

    keep_lane: str = "keep_lane"
    slow_down: str = "slow_down"
    change_lane_left: str = "change_lane_left"
    change_lane_right: str = "change_lane_right"


class Controllers:
    """Handles vehicle controller selection."""

    @staticmethod
    def perform_action(
        sim,
        agent_id,
        vehicle,
        action,
        controller_state,
        sensor_state,
        action_space,
    ):
        """Calls control for the given vehicle based on a given action space and action.

        Args:
            sim:
                A simulation instance.
            agent_id:
                An agent within the simulation that is associated with a vehicle.
            vehicle:
                A vehicle within the simulation that is associated with an agent.
            action:
                The action for the controller to perform.
            controller_state:
                The last vehicle controller state as relates to its action space.
            sensor_state:
                The state of a vehicle sensor as relates to vehicle sensors.
            action_space:
                The action space of the provided action.
        """
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
                vehicle, action, controller_state, dt_sec=sim.last_dt
            )
        elif action_space == ActionSpaceType.Trajectory:
            TrajectoryTrackingController.perform_trajectory_tracking_PD(
                action,
                vehicle,
                controller_state,
                dt_sec=sim.last_dt,
            )
        elif action_space == ActionSpaceType.MPC:
            TrajectoryTrackingController.perform_trajectory_tracking_MPC(
                action, vehicle, controller_state, sim.last_dt
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

            action = LaneAction(value=action)
            # 12.5 m/s (45 km/h) is used as the nominal speed for lane change.
            # For keep_lane, the nominal speed is set to 15 m/s (54 km/h).
            if action == LaneAction.keep_lane:
                perform_lane_following(target_speed=15, lane_change=0)
            elif action == LaneAction.slow_down:
                perform_lane_following(target_speed=0, lane_change=0)
            elif action == LaneAction.change_lane_left:
                perform_lane_following(target_speed=12.5, lane_change=1)
            elif action == LaneAction.change_lane_right:
                perform_lane_following(target_speed=12.5, lane_change=-1)
        elif action_space == ActionSpaceType.Direct:
            DirectController.perform_action(sim.last_dt, vehicle, action)
        elif action_space in (
            ActionSpaceType.TargetPose,
            ActionSpaceType.MultiTargetPose,
            ActionSpaceType.RelativeTargetPose,
        ):
            motion_action = action
            if action_space is ActionSpaceType.RelativeTargetPose:
                position, heading = vehicle.pose.position, vehicle.pose.heading
                motion_action = [
                    action[0] + position[0],
                    action[1] + position[1],
                    action[2] + heading,
                    0.1,
                ]
            MotionPlannerController.perform_action(
                controller_state, sim.last_dt, vehicle, motion_action
            )
        elif action_space == ActionSpaceType.TrajectoryWithTime:
            TrajectoryInterpolationController.perform_action(
                sim.last_dt, vehicle, action
            )
        else:
            raise ValueError(
                f"perform_action({action_space=}, ...) has failed " "inside controller"
            )

    @staticmethod
    def get_action_shape(action_space: ActionSpaceType):
        """Describes the shape of actions that are used for standard controllers.

        Args:
            action_space (ActionSpaceType): The action space to describe.

        Raises:
            NotImplementedError: The action space requested does is not yet implemented.

        Returns:
            Tuple[Any, str]: The action space and the descriptive attribute.
        """
        # TODO MTA: test the action shapes against dummy agents.
        if action_space == ActionSpaceType.Empty:
            return Union[type(None), Literal[False], Tuple], "null"

        if action_space == ActionSpaceType.Lane:
            return (
                Literal[
                    LaneAction.keep_lane,
                    LaneAction.slow_down,
                    LaneAction.change_lane_left,
                    LaneAction.change_lane_right,
                ],
                "lane_action",
            )

        if action_space in (
            ActionSpaceType.ActuatorDynamic,
            ActionSpaceType.Continuous,
        ):
            return Tuple[float, float, float], ("throttle", "break", "steering")

        if action_space == ActionSpaceType.LaneWithContinuousSpeed:
            return Tuple[float, int], ("lane_speed", "lane_change_delta")

        if action_space in (ActionSpaceType.MPC, ActionSpaceType.Trajectory):
            return Tuple[
                Sequence[float], Sequence[float], Sequence[float], Sequence[float]
            ], ("x_coords", "y_coords", "headings", "speeds")

        if action_space == ActionSpaceType.Direct:
            return Union[float, Tuple[float, float]], [
                "speed",
                ("linear_acceleration", "angular_velocity"),
            ]

        if action_space == ActionSpaceType.TrajectoryWithTime:
            return Tuple[
                Sequence[float],
                Sequence[float],
                Sequence[float],
                Sequence[float],
                Sequence[float],
            ], ("times", "x_coords", "y_coords", "headings", "speeds")

        TargetPoseSpace = Tuple[float, float, float, float]
        TargetPoseAttributes = ("x_coord", "y_coord", "heading", "time_delta")
        if action_space == ActionSpaceType.TargetPose:
            return TargetPoseSpace, TargetPoseAttributes

        if action_space == ActionSpaceType.MultiTargetPose:
            return Dict[str, TargetPoseSpace], {"agent_id": TargetPoseAttributes}

        if action_space == ActionSpaceType.RelativeTargetPose:
            return Tuple[float, float, float], ("delta_x", "delta_y", "delta_heading")

        raise NotImplementedError(f"Type {action_space} is not implemented")


class ControllerOutOfLaneException(Exception):
    """Represents an error due to a vehicle straying too far from any available lane."""

    pass


class ControllerState:
    """Controller state"""

    @staticmethod
    def from_action_space(action_space, vehicle_pose, sim):
        """Generate the appropriate controller state given an action space."""
        if action_space in (
            ActionSpaceType.Lane,
            ActionSpaceType.LaneWithContinuousSpeed,
        ):
            target_lane = sim.road_map.nearest_lane(vehicle_pose.point)
            if not target_lane:
                # This likely means this is a traffic history vehicle that is out-of-lane.
                # If not, maybe increase radius in nearest_lane call?
                raise ControllerOutOfLaneException(
                    "Controller has failed because actor is too far from lane for lane-following."
                )
            return LaneFollowingControllerState(target_lane.lane_id)

        if action_space == ActionSpaceType.ActuatorDynamic:
            return ActuatorDynamicControllerState()

        if action_space == ActionSpaceType.Trajectory:
            return TrajectoryTrackingControllerState()

        if action_space == ActionSpaceType.MPC:
            return TrajectoryTrackingControllerState()

        if action_space in (
            ActionSpaceType.TargetPose,
            ActionSpaceType.MultiTargetPose,
            ActionSpaceType.RelativeTargetPose,
        ):
            return MotionPlannerControllerState()

        # Other action spaces do not need a controller state object
        return None
