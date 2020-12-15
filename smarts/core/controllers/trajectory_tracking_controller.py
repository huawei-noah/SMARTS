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

METER_PER_SECOND_TO_KM_PER_HR = 3.6


class TrajectoryTrackingControllerState:
    def __init__(self):
        self.heading_error_gain = None
        self.lateral_error_gain = None
        self.heading_error = 0
        self.lateral_error = 0
        self.velocity_error = 0
        self.integral_velocity_error = 0
        self.integral_windup_error = 0
        self.steering_state = 0
        self.throttle_state = 0


class TrajectoryTrackingController:
    @staticmethod
    def perform_trajectory_tracking_MPC(
        trajectory, vehicle, state, dt_sec, prediction_horizon=5
    ):
        half_vehicle_len = vehicle.length / 2
        vehicle_mass, vehicle_inertia_z = vehicle.chassis.mass_and_inertia
        (
            heading_error,
            lateral_error,
        ) = TrajectoryTrackingController.calulate_heading_lateral_error(
            vehicle=vehicle,
            trajectory=trajectory,
            initial_look_ahead_distant=3,
            speed_reduction_activation=True,
        )

        raw_throttle = TrajectoryTrackingController.calculate_raw_throttle_feedback(
            vehicle=vehicle,
            state=state,
            trajectory=trajectory,
            velocity_gain=1,
            velocity_integral_gain=0,
            integral_velocity_error=0,
            velocity_damping_gain=0,
            windup_gain=0,
            traction_gain=8,
            speed_reduction_activation=True,
            throttle_filter_constant=10,
            dt_sec=dt_sec,
        )[0]
        if raw_throttle > 0:
            brake_norm, throttle_norm = 0, np.clip(raw_throttle, 0, 1)
        else:
            brake_norm, throttle_norm = np.clip(-raw_throttle, 0, 1), 0
        longitudinal_velocity = vehicle.chassis.longitudinal_lateral_speed[0]
        # If longitudinal speed is less than 0.1 (m/s) then use 0.1 (m/s) as
        # the speed for state and input matrix calculations.
        longitudinal_velocity = max(0.1, longitudinal_velocity)

        state_matrix = np.array(
            [
                [0, 1, 0, 0],
                [
                    0,
                    -(
                        vehicle.chassis.front_rear_stiffness[0]
                        + vehicle.chassis.front_rear_stiffness[1]
                    )
                    / (vehicle_mass * longitudinal_velocity),
                    (
                        vehicle.chassis.front_rear_stiffness[0]
                        + vehicle.chassis.front_rear_stiffness[1]
                    )
                    / vehicle_mass,
                    half_vehicle_len
                    * (
                        vehicle.chassis.front_rear_stiffness[0]
                        + vehicle.chassis.front_rear_stiffness[1]
                    )
                    / (vehicle_mass * longitudinal_velocity),
                ],
                [0, 0, 0, 1],
                [
                    0,
                    half_vehicle_len
                    * (
                        -vehicle.chassis.front_rear_stiffness[0]
                        + vehicle.chassis.front_rear_stiffness[1]
                    )
                    / (vehicle_mass * longitudinal_velocity),
                    half_vehicle_len
                    * (
                        vehicle.chassis.front_rear_stiffness[0]
                        - vehicle.chassis.front_rear_stiffness[1]
                    )
                    / vehicle_mass,
                    (half_vehicle_len ** 2)
                    * (
                        vehicle.chassis.front_rear_stiffness[0]
                        - vehicle.chassis.front_rear_stiffness[1]
                    )
                    / (vehicle_mass * longitudinal_velocity),
                ],
            ]
        )
        input_matrix = np.array(
            [
                [0],
                [vehicle.chassis.front_rear_stiffness[0] / vehicle_mass],
                [0],
                [vehicle.chassis.front_rear_stiffness[1] / vehicle_inertia_z],
            ]
        )
        drift_matrix = TrajectoryTrackingController.mpc_drift_matrix(
            vehicle, trajectory, prediction_horizon
        )
        steering_norm = -TrajectoryTrackingController.MPC(
            trajectory,
            heading_error,
            lateral_error,
            dt_sec,
            state_matrix,
            input_matrix,
            drift_matrix,
            prediction_horizon,
        )

        vehicle.control(
            throttle=throttle_norm, brake=brake_norm, steering=steering_norm,
        )

    # Final values are the gains at 80 km/hr (22.2 m/s).
    @staticmethod
    def perform_trajectory_tracking_PD(
        trajectory, vehicle, state, dt_sec,
    ):
        # Controller parameters for trajectory tracking.
        params = vehicle.chassis.controller_parameters
        final_heading_gain = params["final_heading_gain"]
        final_lateral_gain = params["final_lateral_gain"]
        final_steering_filter_constant = params["final_steering_filter_constant"]
        throttle_filter_constant = params["throttle_filter_constant"]
        velocity_gain = params["velocity_gain"]
        velocity_integral_gain = params["velocity_integral_gain"]
        traction_gain = params["traction_gain"]
        final_lateral_error_derivative_gain = params[
            "final_lateral_error_derivative_gain"
        ]
        final_heading_error_derivative_gain = params[
            "final_heading_error_derivative_gain"
        ]
        initial_look_ahead_distant = params["initial_look_ahead_distant"]
        derivative_activation = params["derivative_activation"]
        speed_reduction_activation = params["speed_reduction_activation"]
        velocity_damping_gain = params["velocity_damping_gain"]
        windup_gain = params["windup_gain"]

        curvature_radius = TrajectoryTrackingController.curvature_calculation(
            trajectory
        )
        # The gains are varying according to the desired velocity along
        # the trajectory. To achieve this, the desired speed is normalized
        # between 20 (km/hr) to 80 (km/hr) and the respective gains are
        # calculated using interpolation. 3, 0.03, 1.5, 0.2 are the
        # controller gains for lateral error, heading error and their
        #  derivatives at the desired speed 20 (km/hr).
        normalized_speed = np.clip(
            (METER_PER_SECOND_TO_KM_PER_HR * trajectory[3][0] - 20) / (80 - 20), 0, 1
        )
        lateral_gain = lerp(3, final_lateral_gain, normalized_speed)
        heading_gain = lerp(0.03, final_heading_gain, normalized_speed)
        steering_filter_constant = lerp(
            2, final_steering_filter_constant, normalized_speed
        )
        heading_error_derivative_gain = lerp(
            1.5, final_heading_error_derivative_gain, normalized_speed
        )
        lateral_error_derivative_gain = lerp(
            0.2, final_lateral_error_derivative_gain, normalized_speed
        )

        (
            heading_error,
            lateral_error,
        ) = TrajectoryTrackingController.calulate_heading_lateral_error(
            vehicle, trajectory, initial_look_ahead_distant, speed_reduction_activation
        )

        # Derivative terms of the controller (use with caution for large time steps>=0.1).
        # Increasing the values will increase the convergence time and reduces the oscillation.
        derivative_term = (
            +heading_error_derivative_gain * vehicle.chassis.yaw_rate[2]
            + lateral_error_derivative_gain
            * (lateral_error - state.lateral_error)
            / dt_sec
        )
        # Raw steering controller, default values 0.11 and 0.65 are used for heading and
        # lateral control gains.
        # TODO: The lateral and heading gains of the steering controller should be
        # calculated based on the current velocity. The coefficient value for the
        # feed forward term is 0.1 and it depends on the cornering stifness and
        # vehicle inertia properties.
        steering_feed_forward_term = 0.1 * (1 / curvature_radius) * (vehicle.speed) ** 2
        steering_raw = np.clip(
            derivative_activation * derivative_term
            + math.degrees(heading_gain * (heading_error))
            + 1 * lateral_gain * lateral_error
            - steering_feed_forward_term,
            -1,
            1,
        )
        # The steering linear low pass filter.
        state.steering_state = low_pass_filter(
            steering_raw, state.steering_state, steering_filter_constant, dt_sec
        )

        (
            raw_throttle,
            desired_speed,
        ) = TrajectoryTrackingController.calculate_raw_throttle_feedback(
            vehicle,
            state,
            trajectory,
            velocity_gain,
            velocity_integral_gain,
            state.integral_velocity_error,
            velocity_damping_gain,
            windup_gain,
            traction_gain,
            speed_reduction_activation,
            throttle_filter_constant,
            dt_sec,
        )

        if raw_throttle > 0:
            brake_norm, throttle_norm = 0, np.clip(raw_throttle, 0, 1)
        else:
            brake_norm, throttle_norm = np.clip(-raw_throttle, 0, 1), 0

        state.heading_error = heading_error
        state.lateral_error = lateral_error
        state.integral_velocity_error += (vehicle.speed - desired_speed) * dt_sec

        vehicle.control(
            throttle=throttle_norm, brake=brake_norm, steering=state.steering_state,
        )

    @staticmethod
    def calculate_raw_throttle_feedback(
        vehicle,
        state,
        trajectory,
        velocity_gain,
        velocity_integral_gain,
        integral_velocity_error,
        velocity_damping_gain,
        windup_gain,
        traction_gain,
        speed_reduction_activation,
        throttle_filter_constant,
        dt_sec,
    ):
        desired_speed = trajectory[3][0]
        # If the vehicle is on the curvy portion of the road, then the desired speed
        # will be reduced to 80 percent of the desired speed of the trajectory.
        # Value 4 is the number of ahead trajectory points for starting the calculation
        # of the radius of curvature. Value 100 is the threshold for reducing the
        # desired speed profile. This value can be approximated using the
        # formula v^2/R=mu*g. In addition, we used additional threshold 30 for very
        # sharp turn in which the desired speed of the vehicle is set to 80
        # percent of the original values with the upperbound of 29.8 m/s(30 km/hr).
        absolute_ahead_curvature = abs(
            TrajectoryTrackingController.curvature_calculation(trajectory, 4)
        )
        if absolute_ahead_curvature < 30 and speed_reduction_activation:
            desired_speed = np.clip(0.8 * desired_speed, 0, 8.3)
        elif absolute_ahead_curvature < 100 and speed_reduction_activation:
            desired_speed *= 0.8

        # Main velocity profile tracking controller, the default gain value is 0.1.
        # Default value 3 for the traction controller term involving lateral velocity is
        # chosen for better performance on curvy portion of the road. Note that
        # it should be at least one order of magnitude above the velocity
        # tracking term to ensure stability.
        velocity_error = vehicle.speed - desired_speed
        velocity_error_damping_term = (velocity_error - state.velocity_error) / dt_sec
        raw_throttle = METER_PER_SECOND_TO_KM_PER_HR * (
            -1 * velocity_gain * velocity_error
            - velocity_integral_gain
            * (integral_velocity_error + windup_gain * state.integral_windup_error)
            - velocity_damping_gain * velocity_error_damping_term
        )
        state.velocity_error = velocity_error
        # The anti-windup term inspired by:
        # https://ocw.mit.edu/courses/aeronautics-and-astronautics/16-30-feedback-control-systems-fall-2010/lecture-notes/MIT16_30F10_lec23.pdf
        state.integral_windup_error = np.clip(raw_throttle, -1, 1) - raw_throttle

        # The low pass filter for throttle. The traction control term is
        # applied after filter is applied to the velocity feedback.
        state.throttle_state = low_pass_filter(
            raw_throttle,
            state.throttle_state,
            throttle_filter_constant,
            dt_sec,
            raw_value=-traction_gain
            * abs(vehicle.chassis.longitudinal_lateral_speed[1]),
        )

        return (state.throttle_state, desired_speed)

    @staticmethod
    def calulate_heading_lateral_error(
        vehicle, trajectory, initial_look_ahead_distant, speed_reduction_activation
    ):
        heading_error = min_angles_difference_signed(
            (vehicle.heading % (2 * math.pi)), trajectory[2][0]
        )

        # Number of look ahead points to calculate the
        # look ahead error.
        # TODO: Find the number of points using the
        # distance between trajectory points.
        look_ahead_points = initial_look_ahead_distant
        # If we are on the curvy portion of the road
        # we need to decrease the values for look ahead calculation
        # The value 30 for road curviness is obtained empirically
        # for bullet model.
        if (
            abs(TrajectoryTrackingController.curvature_calculation(trajectory, 4)) < 30
            and speed_reduction_activation
        ):
            initial_look_ahead_distant = 1
            look_ahead_points = 1

        path_vector = radians_to_vec(
            trajectory[2][min([look_ahead_points, len(trajectory[2]) - 1])]
        )

        vehicle_look_ahead_pt = [
            vehicle.position[0]
            - initial_look_ahead_distant * math.sin(vehicle.heading),
            vehicle.position[1]
            + initial_look_ahead_distant * math.cos(vehicle.heading),
        ]

        lateral_error = signed_dist_to_line(
            vehicle_look_ahead_pt,
            [
                trajectory[0][min([look_ahead_points, len(trajectory[0]) - 1])],
                trajectory[1][min([look_ahead_points, len(trajectory[1]) - 1])],
            ],
            path_vector,
        )
        return (heading_error, lateral_error)

    @staticmethod
    def curvature_calculation(trajectory, offset=0):
        number_ahead_points = 5
        relative_heading_sum, relative_distant_sum = 0, 0

        if len(trajectory[2]) <= number_ahead_points + offset:
            return 1e20

        for i in range(number_ahead_points):
            relative_heading_temp = min_angles_difference_signed(
                trajectory[2][i + 1 + offset], trajectory[2][i + offset]
            )
            relative_heading_sum += relative_heading_temp
            relative_distant_sum += abs(
                math.sqrt(
                    (trajectory[0][i + offset] - trajectory[0][i + offset + 1]) ** 2
                    + (trajectory[1][i + offset] - trajectory[1][i + offset + 1]) ** 2
                )
            )
        # If relative_heading_sum is zero, then the local radius
        # of curvature is infinite, i.e. the local trajectory is
        # similar to a straight line.
        if relative_heading_sum == 0:
            return 1e20
        else:
            curvature_radius = relative_distant_sum / relative_heading_sum

        if curvature_radius == 0:
            curvature_radius == 1e-2
        return curvature_radius

    @staticmethod
    def mpc_drift_matrix(vehicle, trajectory, prediction_horizon=1):
        half_vehicle_len = vehicle.length / 2
        vehicle_mass, vehicle_inertia_z = vehicle.chassis.mass_and_inertia
        factor_matrix = np.array(
            [
                [0],
                [
                    (
                        half_vehicle_len * vehicle.chassis.front_rear_stiffness[0]
                        + half_vehicle_len * vehicle.chassis.front_rear_stiffness[1]
                    )
                    / vehicle_mass
                    - (vehicle.chassis.longitudinal_lateral_speed[0]) ** 2
                ],
                [0],
                [
                    (
                        (half_vehicle_len ** 2)
                        * vehicle.chassis.front_rear_stiffness[0]
                        - (half_vehicle_len ** 2)
                        * vehicle.chassis.front_rear_stiffness[1]
                    )
                    / vehicle_inertia_z
                ],
            ]
        )
        matrix_drift = (
            TrajectoryTrackingController.curvature_calculation(trajectory) ** -1
        ) * factor_matrix
        for i in range(prediction_horizon - 1):
            matrix_drift = np.concatenate(
                (
                    matrix_drift,
                    (
                        TrajectoryTrackingController.curvature_calculation(
                            trajectory, i
                        )
                        ** -1
                    )
                    * factor_matrix,
                ),
                axis=1,
            )

        return matrix_drift

    @staticmethod
    def MPC(
        trajectory,
        heading_error,
        lateral_error,
        dt,
        state_matrix,
        input_matrix,
        drift_matrix,
        prediction_horizon=5,
    ):
        # Implementation of MPC, please see the following ref:
        # Convex Optimization – Boyd and Vandenberghe
        # https://markcannon.github.io/assets/downloads/teaching/C21_Model_Predictive_Control/mpc_notes.pdf
        matrix_A = np.eye(state_matrix.shape[0]) + dt * state_matrix
        matrix_B = dt * input_matrix
        matrix_B0 = dt * drift_matrix
        matrix_M = matrix_A
        matrix_C = np.concatenate(
            (
                matrix_B,
                np.zeros(
                    [matrix_B.shape[0], (prediction_horizon - 1) * matrix_B.shape[1]]
                ),
            ),
            axis=1,
        )

        matrix_T_tilde = matrix_B0[:, 0]

        for i in range(prediction_horizon - 1):
            matrix_M = np.concatenate((matrix_M, matrix_power(matrix_A, i + 2)), axis=0)
            matrix_T_tilde = np.concatenate(
                (
                    matrix_T_tilde,
                    np.matmul(
                        matrix_A,
                        matrix_T_tilde[
                            matrix_T_tilde.shape[0]
                            - matrix_B0.shape[0] : matrix_T_tilde.shape[0]
                        ],
                        matrix_B0[:, i + 1],
                    ),
                ),
                axis=0,
            )
            temp = np.matmul(matrix_power(matrix_A, i + 1), matrix_B)
            for j in range(i + 1, 0, -1):
                temp = np.concatenate(
                    (temp, np.matmul(matrix_power(matrix_A, j - 1), matrix_B)), axis=1
                )
            temp = np.concatenate(
                (
                    temp,
                    np.zeros(
                        [
                            matrix_B.shape[0],
                            (prediction_horizon - i - 2) * matrix_B.shape[1],
                        ]
                    ),
                ),
                axis=1,
            )
            matrix_C = np.concatenate((matrix_C, temp), axis=0)

        # Q_tilde contains the MPC state cost weights. The ordering of
        # gains are as [lateral_error,lateral_velocity,heading_error,yaw_rate].
        # Increasing the lateral_error weight can cause oscillations which can
        # be damped out by increasing yaw_rate weight.
        Q_tilde = np.kron(np.eye(prediction_horizon), 0.1 * np.diag([354, 0, 14, 250]))
        # R_tilde contain the steering input weight.
        R_tilde = np.kron(np.eye(prediction_horizon), np.eye(1))
        matrix_H = (np.transpose(matrix_C)).dot(Q_tilde).dot(matrix_C) + R_tilde
        matrix_F = (np.transpose(matrix_C)).dot(Q_tilde).dot(matrix_M)

        matrix_F1 = (np.transpose(matrix_C)).dot(Q_tilde).dot(matrix_T_tilde)

        # This is the solution to the unconstrained optimization problem
        # for the MPC.
        unconstrained_optimal_solution = np.matmul(
            np.linalg.inv(2 * matrix_H),
            np.matmul(matrix_F, np.array([[lateral_error], [0], [heading_error], [0]]))
            + np.reshape(matrix_F1, (-1, 1)),
        )[0][0]

        return np.clip(-unconstrained_optimal_solution, -1, 1)
