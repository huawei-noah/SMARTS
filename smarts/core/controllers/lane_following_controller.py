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
from smarts.core.controllers.trajectory_tracking_controller import (
    TrajectoryTrackingControllerState,
    TrajectoryTrackingController,
)
from smarts.core.chassis import AckermannChassis
from smarts.core.utils.math import (
    lerp,
    radians_to_vec,
    signed_dist_to_line,
    min_angles_difference_signed,
    low_pass_filter,
)

METER_PER_SECOND_TO_KM_PER_HR = 3.6


class LaneFollowingControllerState:
    # TODO: Consider making this immutable and making `LaneFollowingController`
    #       generate new state object.
    def __init__(self, target_lane_id):
        self.target_lane_id = target_lane_id
        self.target_speed = None
        self.heading_error_gain = None
        self.lateral_error_gain = None
        self.lateral_integral_error = 0
        self.steering_state = 0
        self.throttle_state = 0
        self.min_curvature_location = (None, None)


class LaneFollowingController:
    lateral_error = -35
    heading_error = -15
    yaw_rate = -2
    side_slip_angle = -3

    @classmethod
    def perform_lane_following(
        cls,
        sim,
        agent_id,
        vehicle,
        controller_state,
        sensor_state,
        target_speed=12.5,
        lane_change=0,
    ):
        assert isinstance(vehicle.chassis, AckermannChassis)
        state = controller_state
        # This lookahead value is coupled with a few calculations below, changing it
        # may affect stability of the controller.
        wp_paths = sensor_state.mission_planner.waypoint_paths_at(
            vehicle.pose, lookahead=30
        )
        current_lane = LaneFollowingController.find_current_lane(
            wp_paths, vehicle.position
        )
        wp_path = wp_paths[np.clip(current_lane + lane_change, 0, len(wp_paths) - 1)]

        # we compute a road "curviness" to inform our throttle activation.
        # We should move slowly when we are on curvy roads.
        ewma_road_curviness = 0.0
        for wp_a, wp_b in reversed(list(zip(wp_path, wp_path[1:]))):
            ewma_road_curviness = lerp(
                ewma_road_curviness,
                math.degrees(abs(wp_a.relative_heading(wp_b.heading))),
                0.03,
            )

        road_curviness_normalization = 2.5
        road_curviness = np.clip(
            ewma_road_curviness / road_curviness_normalization, 0, 1
        )
        # Number of trajectory point used for curvature calculation.
        num_trajectory_points = min([10, len(wp_path)])
        trajectory = [
            [wp_path[i].pos[0] for i in range(num_trajectory_points)],
            [wp_path[i].pos[1] for i in range(num_trajectory_points)],
            [wp_path[i].heading for i in range(num_trajectory_points)],
        ]
        # The following calculates the radius of curvature for the 4th
        # waypoints in the waypoint list. Value 4 is chosen to ensure
        # that the heading error correction is triggered before the vehicle
        # reaches to a sharp turn defined be min_curvature.
        look_ahead_curvature = abs(
            TrajectoryTrackingController.curvature_calculation(trajectory, 4)
        )
        # Minimum curvature limit for pushing forward the waypoint
        # which is used for heading error calculation.
        min_curvature = 2
        # If the look_ahead_curvature is less than the min_curvature, then
        # update the location of the points which its curvature is less than
        # min_curvature.
        if look_ahead_curvature <= min_curvature:
            state.min_curvature_location = (wp_path[4].pos[0], wp_path[4].pos[1])

        # LOOK AHEAD ERROR SETTING
        # look_ahead_wp_num is the ahead waypoint which is used to
        # calculate the lateral error TODO: use adaptive setting to
        # choose the ahead waypoint

        # if the road_curviness is high(i.e. > 0.5), we reduce the
        # look_ahead_wp_num to calculate a more accurate lateral error
        # normal look ahead distant is set to 8 meters which is reduced
        # to 6 meters when the curvature increases.
        # Note: waypoints are spaced at roughly 1 meter apart
        if road_curviness > 0.5:
            look_ahead_wp_num = 3
        else:
            look_ahead_wp_num = 4

        look_ahead_wp_num = min(look_ahead_wp_num, len(wp_path) - 1)

        reference_heading = wp_path[0].heading
        look_ahead_wp = wp_path[look_ahead_wp_num]
        look_ahead_dist = look_ahead_wp.dist_to(vehicle.position)
        vehicle_look_ahead_pt = [
            vehicle.position[0] - look_ahead_dist * math.sin(vehicle.heading),
            vehicle.position[1] + look_ahead_dist * math.cos(vehicle.heading),
        ]

        # 5.56 m/s (20 km/h), 6.94 m/s (25 km/h) are desired speed for different thresholds
        #   for road curviness
        # 0.5 , 0.8 are dimensionless thresholds for road_curviness.
        # 0.8 and 0.6 are the longitudinal velocity controller
        # proportional gains for different road curvinesss.
        if road_curviness < 0.5:
            raw_throttle = (
                -METER_PER_SECOND_TO_KM_PER_HR * 0.8 * (vehicle.speed - target_speed)
            )
        elif road_curviness > 0.5 and road_curviness < 0.8:
            raw_throttle = (
                -0.6
                * METER_PER_SECOND_TO_KM_PER_HR
                * (vehicle.speed - np.clip(target_speed, 0, 6.94))
            )
        else:
            raw_throttle = (
                -0.6
                * METER_PER_SECOND_TO_KM_PER_HR
                * (vehicle.speed - np.clip(target_speed, 0, 5.56))
            )
        # If the distance of the vehicle to the ahead point for which
        # the waypoint curvature is less than min_curvature is less than
        # 2 meters, then push forward the waypoint which is used to
        # calculate the heading error.
        if (state.min_curvature_location != (None, None)) and math.sqrt(
            (vehicle.position[0] - state.min_curvature_location[0]) ** 2
            + (vehicle.position[1] - state.min_curvature_location[1]) ** 2
        ) < 2:
            reference_heading = wp_path[look_ahead_wp_num].heading

        # Desired closed loop poles of the lateral dynamics
        # The higher the absolute value, the closed loop response will
        # be faster for that state, the four states of that are used for
        # Linearization of the lateral dynamics are:
        # [lateral error, heading error, yaw_rate, side_slip angle]
        desired_poles = np.array(
            [cls.lateral_error, cls.heading_error, cls.yaw_rate, cls.side_slip_angle,]
        )

        LaneFollowingController.calculate_lateral_gains(
            sim, state, vehicle, desired_poles, target_speed
        )
        # LOOK AHEAD CONTROLLER
        controller_lat_error = wp_path[look_ahead_wp_num].signed_lateral_error(
            vehicle_look_ahead_pt
        )

        abs_heading_error = min(
            abs((vehicle.heading % (2 * math.pi)) - reference_heading),
            abs(
                2 * math.pi - abs((vehicle.heading % (2 * math.pi)) - reference_heading)
            ),
        )

        brake_norm = 0
        if raw_throttle < 0:
            brake_norm = np.clip(-raw_throttle, 0, 1)
            throttle_norm = 0
        else:
            # The term involving absolute value of the lateral speed is
            # added as a traction control strategy, The traction controller
            # gain is set to 4.5, the lower the value, the vehicle becomes
            # more agile but may result in instability in harsh curves
            # with high speeds.
            throttle_norm = np.clip(
                raw_throttle
                - 4.5
                * METER_PER_SECOND_TO_KM_PER_HR
                * abs(vehicle.chassis.longitudinal_lateral_speed[1]),
                0,
                1,
            )
        # The feedback term involving yaw rate is added to reduce
        # the oscillation in vehicle heading, the proportional gain for
        # yaw rate is set to 2.75, the higher value results in less
        # oscillation in heading angle. 0.3 is the integral controller
        # gain for lateral error. The feedforward term based on the
        # curvature is added to enhance the transient performance when
        # the road curvature changes locally.
        state.lateral_integral_error += sim.timestep_sec * controller_lat_error
        curvature_radius = TrajectoryTrackingController.curvature_calculation(
            trajectory
        )
        # The feed forward term for the  steering controller. This
        # term is proportionate to Ux^2/R. The coefficient 0.15 is
        # chosen to enhance the transient tracking performance.
        # This coefficient also depends on the inertia properties
        # and the cornering stiffness of the tires. See:
        # https://www.tandfonline.com/doi/full/10.1080/00423114.2015.1055279
        steering_controller_feed_forward = (
            1 * 0.15 * (1 / curvature_radius) * (vehicle.speed) ** 2
        )
        steering_norm = np.clip(
            -1
            * math.degrees(state.heading_error_gain)
            * (
                abs_heading_error
                * np.sign(reference_heading - (vehicle.heading % (2 * math.pi)))
            )
            + 1 * state.lateral_error_gain * (controller_lat_error)
            + 2.75 * vehicle.chassis.yaw_rate[2]
            + 0.3 * state.lateral_integral_error
            - steering_controller_feed_forward,
            -1,
            1,
        )
        # The steering low pass filter, 5.5 is the constant of the
        # first order linear low pass filter.
        steering_filter_constant = 5.5

        state.steering_state = low_pass_filter(
            steering_norm,
            state.steering_state,
            steering_filter_constant,
            sim.timestep_sec,
        )

        # The Throttle low pass filter, 2 is the constant of the
        # first order linear low pass filter.
        # TODO: Add low pass filter for brake.
        throttle_filter_constant = 2

        state.throttle_state = low_pass_filter(
            throttle_norm,
            state.throttle_state,
            throttle_filter_constant,
            sim.timestep_sec,
            lower_bound=0,
        )
        # Applying control actions to the vehicle
        vehicle.control(
            throttle=state.throttle_state,
            brake=brake_norm,
            steering=state.steering_state,
        )

        LaneFollowingController._update_target_lane_if_reached_end_of_lane(
            agent_id, vehicle, controller_state, sensor_state
        )

    @staticmethod
    def find_current_lane(wp_paths, vehicle_position):
        relative_distant_lane = [
            np.linalg.norm(wp_paths[idx][0].pos - vehicle_position[0:2])
            for idx in range(len(wp_paths))
        ]
        return np.argmin(relative_distant_lane)

    @staticmethod
    def calculate_lateral_gains(sim, state, vehicle, desired_poles, target_speed):
        # Only calculate gains if the target_speed is updated.
        # TODO: Replace this w/ an isclose(...) check
        if state.target_speed == target_speed:
            return

        state.target_speed = target_speed

        # Vehicle params
        half_vehicle_len = vehicle.length / 2
        vehicle_mass, vehicle_inertia_z = vehicle.chassis.mass_and_inertia
        road_stiffness = sim.road_stiffness

        # Linearization of lateral dynamics
        if target_speed > 0:
            state_matrix = np.array(
                [
                    [0, target_speed, 0, target_speed],
                    [0, 0, 1, 0],
                    [
                        0,
                        0,
                        -(2 * road_stiffness * (half_vehicle_len ** 2))
                        / (target_speed * vehicle_inertia_z),
                        0,
                    ],
                    [0, 0, -1, -2 * road_stiffness / (vehicle_mass * target_speed),],
                ]
            )
            input_matrix = np.array(
                [
                    [0],
                    [0],
                    [half_vehicle_len * road_stiffness / vehicle_inertia_z],
                    [road_stiffness / (vehicle_mass * target_speed)],
                ]
            )
            fsf1 = signal.place_poles(
                state_matrix, input_matrix, desired_poles, method="KNV0"
            )
            # 0.01 and 0.015 denote the max and min gains for heading controller
            # This is done to ensure that the linearization error will not affect
            # the stability of the controller.
            state.heading_error_gain = np.clip(fsf1.gain_matrix[0][1], 0.02, 0.04)
            # 3.4 and 4.1 denote the max and min gains for lateral error controller
            # As for heading, this is done to ensure that the linearization error
            # will not affect the stability and performance of the controller.
            state.lateral_error_gain = np.clip(fsf1.gain_matrix[0][0], 3.4, 4.1)

        else:
            # 0.01 and 0.36 are initial values for heading and lateral gains
            # This is only done to ensure that the vehicle starts to move for
            # the first time step where speed=0
            state.heading_error_gain = 0.01
            state.lateral_error_gain = 0.36

    @staticmethod
    def _update_target_lane_if_reached_end_of_lane(
        agent_id, vehicle, controller_state, sensor_state
    ):
        # When we reach the end of our target lane, we need to update it
        # to the next lane best lane along the path
        state = controller_state
        paths = sensor_state.mission_planner.waypoint_paths_on_lane_at(
            vehicle.pose, state.target_lane_id, lookahead=2
        )

        candidate_next_wps = []
        for path in paths:
            wps_of_next_lanes_on_path = [
                wp for wp in path if wp.lane_id != state.target_lane_id
            ]

            if wps_of_next_lanes_on_path == []:
                continue

            next_wp = wps_of_next_lanes_on_path[0]
            candidate_next_wps.append(next_wp)

        if candidate_next_wps == []:
            return

        next_wp = min(
            candidate_next_wps,
            key=lambda wp: abs(wp.signed_lateral_error(vehicle.position))
            + abs(wp.relative_heading(vehicle.heading)),
        )

        state.target_lane_id = next_wp.lane_id
