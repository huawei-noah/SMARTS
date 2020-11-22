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
import multiprocessing as mp

import numpy as np


class TireForces:
    def __init__(self, stiffness, road_friction):
        self.c_alpha_front, self.c_alpha_rear, self.c_x_front, self.c_x_rear = stiffness
        self.road_friction = road_friction

    @staticmethod
    def _calculate_tire_angles(chassis, client, action):

        # Currently, we use parallel steering for front steering.
        fl_tire_angle = -chassis.steering + np.pi * 0.5 + chassis.pose.heading
        fr_tire_angle = fl_tire_angle
        rl_tire_angle = np.pi * 0.5 + chassis.pose.heading
        rr_tire_angle = rl_tire_angle
        return [fl_tire_angle, fr_tire_angle, rl_tire_angle, rr_tire_angle]

    @staticmethod
    def _calculate_slip_angles(chassis, client, action):

        steering_angles_corners = [
            -chassis.steering,
            -chassis.steering,
            0,
            0,
        ]
        slip_angles = np.zeros(4)
        relative_corner_vector = [
            np.array(
                [
                    chassis.front_rear_axle_CG_distance[0],
                    0.5 * chassis.front_track_width,
                    0,
                ]
            ),
            np.array(
                [
                    chassis.front_rear_axle_CG_distance[0],
                    -0.5 * chassis.front_track_width,
                    0,
                ]
            ),
            np.array(
                [
                    -chassis.front_rear_axle_CG_distance[1],
                    0.5 * chassis.rear_track_width,
                    0,
                ]
            ),
            np.array(
                [
                    -chassis.front_rear_axle_CG_distance[1],
                    -0.5 * chassis.rear_track_width,
                    0,
                ]
            ),
        ]
        plane_speed_vector = np.array(
            [
                chassis.longitudinal_lateral_speed[0],
                -chassis.longitudinal_lateral_speed[1],
                0,
            ]
        )

        abs_speed_corners = [
            math.sqrt(
                client.getLinkState(chassis.bullet_id, wheel_id, 1)[6][0] ** 2
                + client.getLinkState(chassis.bullet_id, wheel_id, 1)[6][1] ** 2
            )
            for wheel_id in [2, 4, 5, 6]
        ]

        # 0.1 is the threshold for speed of the center of wheel
        # to activate slip angle caluclation.
        for i in range(4):
            if abs(abs_speed_corners[i]) > 0.1:

                corner_speed = plane_speed_vector + np.cross(
                    np.array([0, 0, chassis.yaw_rate[2]]), relative_corner_vector[i]
                )
                slip_angles[i] = steering_angles_corners[i] - (
                    (math.atan2(corner_speed[1], corner_speed[0],))
                )

        return slip_angles

    @staticmethod
    def _calculate_slip_ratios(chassis, client, action):
        slip_ratios = np.zeros(4)
        tires_center_speed = np.zeros(4)
        tire_angles = TireForces._calculate_tire_angles(chassis, client, action)

        wheel_index = [2, 4, 5, 6]

        wheel_spin = [
            math.sqrt(
                client.getLinkState(chassis.bullet_id, wheel_id, 1)[7][0] ** 2
                + client.getLinkState(chassis.bullet_id, wheel_id, 1)[7][1] ** 2
            )
            for wheel_id in [2, 4, 5, 6]
        ]

        for idx in range(len(wheel_index)):

            tires_center_speed[idx] = client.getLinkState(
                chassis.bullet_id, wheel_index[idx], 1
            )[6][0] * math.cos(tire_angles[idx]) + client.getLinkState(
                chassis.bullet_id, wheel_index[idx], 1
            )[
                6
            ][
                1
            ] * math.sin(
                tire_angles[idx]
            )
            if abs(wheel_spin[idx]) >= 0.1:
                slip_ratios[idx] = (
                    chassis.wheel_radius * wheel_spin[idx] - tires_center_speed[idx]
                ) / (chassis.wheel_radius * wheel_spin[idx])

        return slip_ratios

    @staticmethod
    def build_tire_model(stiffness, tire_model_type, road_friction):
        if tire_model_type == "LinearTireforce(wheel)":
            return LinearTireForces(stiffness, road_friction)
        elif tire_model_type == "LinearTireforce(contact)":
            return LinearTireForcesContact(stiffness, road_friction)
        elif tire_model_type == "NonlinearTireForces(wheel)":
            return NonlinearTireForces(stiffness, road_friction)
        elif tire_model_type == "NonlinearTireForces(contact)":
            return NonlinearTireForcesContact(stiffness, road_friction)
        else:
            raise Exception("Requested tire force model does not exist.")


class LinearTireForces(TireForces):
    def _calculate_tire_forces(self, chassis, client, action):

        tire_angles = self._calculate_tire_angles(chassis, client, action)

        (
            fl_slip_angle,
            fr_slip_angle,
            rl_slip_angle,
            rr_slip_angle,
        ) = self._calculate_slip_angles(chassis, client, action)
        # Maximum lateral force for front tires are set to 2000N and
        # for rear tires it is set to 6000 N, this is to ensure stability.
        # Note that the maximum force for rear tires can be calculate using
        # max_lateral_force=sqrt(mu^2*Fn^2-Fx^2), where Fn is the normal force and
        # Fx is the longitudinal force.
        # TODO: Directly get the bounds by using the normal forces from Bullet
        max_normal_force = [2000, 2000, 6000, 6000]
        min_normal_force = [-2000, -2000, -6000, -6000]
        lat_forces = [
            self.road_friction
            * np.clip(
                self.c_alpha_front * fl_slip_angle,
                min_normal_force[0],
                max_normal_force[0],
            ),
            self.road_friction
            * np.clip(
                self.c_alpha_front * fr_slip_angle,
                min_normal_force[1],
                max_normal_force[1],
            ),
            self.road_friction
            * np.clip(
                self.c_alpha_rear * rl_slip_angle,
                min_normal_force[2],
                max_normal_force[2],
            ),
            self.road_friction
            * np.clip(
                self.c_alpha_rear * rr_slip_angle,
                min_normal_force[3],
                max_normal_force[3],
            ),
        ]

        # The maximum force at the center of each wheel is set to 1000N
        lon_forces = [self.road_friction * action[0][idx] * 1000 for idx in range(4)]

        # If brake is activated, then override the throttle. Max brake is set
        # to 100 N,
        # TODO: Replace the maximum brake force using the values of
        # maximum brake torque in chassis.
        if action[1] > 0 and chassis.longitudinal_lateral_speed[0] > 0.1:
            lon_forces = [action[1] * -100 for idx in range(4)]

        forces = []
        for idx in range(4):
            forces.append(
                [
                    lon_forces[idx] * math.cos(tire_angles[idx])
                    + lat_forces[idx] * math.cos(tire_angles[idx] + 0.5 * math.pi),
                    lon_forces[idx] * math.sin(tire_angles[idx])
                    + lat_forces[idx] * math.sin(tire_angles[idx] + 0.5 * math.pi),
                    0,
                ]
            )

        return (forces, lat_forces, lon_forces)

    def apply_tire_forces(self, chassis, client, action):

        wheel_index = [2, 4, 5, 6]
        wheel_positions = [
            np.array(client.getLinkState(chassis.bullet_id, wheel_idx)[0])
            for wheel_idx in wheel_index
        ]

        # Final Bounds for forces at wheels for cases
        # that the rise and fall filters are used at wheels.
        bounds = [15e4, 15e4, 2e5, 2e5]

        forces, lat_forces, lon_forces = self._calculate_tire_forces(
            chassis, client, action
        )

        for idx in range(len(wheel_index)):

            client.applyExternalForce(
                chassis.bullet_id,
                0,
                np.clip(forces[idx], -bounds[idx], bounds[idx]),
                wheel_positions[idx],
                client.WORLD_FRAME,
            )
        return (lat_forces, lon_forces)


# TODO: Implement tire forces at contact points
class LinearTireForcesContact(TireForces):
    pass


# TODO: Implement nonlinear tire forces
class NonlinearTireForces(TireForces):
    pass


# TODO: Implement nonlinear tire forces at contact points
class NonlinearTireForcesContact(TireForces):
    pass
