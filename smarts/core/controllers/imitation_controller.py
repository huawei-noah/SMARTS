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

from smarts.core.chassis import AckermannChassis, BoxChassis
from smarts.core.coordinates import Pose
from smarts.core.utils.math import (
    fast_quaternion_from_angle,
    min_angles_difference_signed,
    radians_to_vec,
    vec_to_radians,
)


class ImitationController:
    @classmethod
    def perform_action(cls, dt, vehicle, action):
        chassis = vehicle.chassis
        if isinstance(action, (int, float)):
            # special case:  setting the initial speed
            if isinstance(chassis, BoxChassis):
                vehicle.control(vehicle.pose, action, dt)
            elif isinstance(chassis, AckermannChassis):
                chassis.speed = action  # hack that calls control internally
            return

        assert isinstance(action, (list, tuple)) and len(action) == 2
        # acceleration is vector in m/s^2, angluar_velocity is scalar in rad/s
        acceleration, angular_velocity = action
        target_heading = (vehicle.heading + angular_velocity * dt) % (2 * math.pi)

        if isinstance(chassis, BoxChassis):
            speed_x, speed_y = radians_to_vec(vehicle.heading) * vehicle.speed
            speed_x += acceleration[0] * dt
            speed_y += acceleration[1] * dt
            new_position = np.array(
                (
                    vehicle.position[0] + dt * speed_x,
                    vehicle.position[1] + dt * speed_y,
                    vehicle.position[2],
                )
            )
            new_pose = Pose(
                orientation=fast_quaternion_from_angle(target_heading),
                position=new_position,
            )
            vehicle.control(new_pose, np.linalg.norm((speed_x, speed_y)), dt)

        elif isinstance(chassis, AckermannChassis):
            acc = np.linalg.norm(acceleration)
            acc_angle = min_angles_difference_signed(
                target_heading, vec_to_radians(acceleration[:2])
            )
            mass = chassis.mass_and_inertia[0]  # in kg
            wheel_radius = chassis.wheel_radius
            if abs(acc_angle) <= math.pi / 2:
                # necessary torque is N*m = kg*m*acceleration
                torque_ratio = mass / (4 * wheel_radius * chassis.max_torque)
                throttle = np.clip(acc * torque_ratio, 0, 1)
                brake = 0
            else:
                throttle = 0
                # necessary torque is N*m = kg*m*acceleration
                torque_ratio = mass / (4 * wheel_radius * chassis.max_btorque)
                brake = np.clip(acc * torque_ratio, 0, 1)

            steering = np.clip(dt * -angular_velocity * chassis.steering_ratio, -1, 1)
            vehicle.control(throttle=throttle, brake=brake, steering=steering)

        else:
            raise Exception("unsupported chassis type")
