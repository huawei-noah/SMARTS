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
from dataclasses import replace

from smarts.core.chassis import AckermannChassis, BoxChassis
from smarts.core.utils.math import fast_quaternion_from_angle, radians_to_vec


class ImitationController:
    @classmethod
    def perform_action(cls, dt, vehicle, action):
        assert len(action) == 2
        # acceleration in m/s^2, angluar_velocity in rad/s
        acceleration, angular_velocity = action

        if isinstance(vehicle.chassis, BoxChassis):
            target_speed = vehicle.speed + acceleration * dt
            target_heading = (vehicle.heading + angular_velocity * dt) % (2 * math.pi)
            speed_x, speed_y = radians_to_vec(target_heading)
            new_position = [
                vehicle.position[0] + dt * speed_x * target_speed,
                vehicle.position[1] + dt * speed_y * target_speed,
                vehicle.position[2],
            ]
            new_pose = replace(
                vehicle.pose,
                orientation=fast_quaternion_from_angle(target_heading),
                position=new_position,
            )
            vehicle.control(new_pose, target_speed)

        # TAI: consider making this a dynamic_action_space, like...
        elif isinstance(vehicle.chassis, AckermannChassis):
            mass = vehicle.chassis.mass_and_inertia[0]  # in kg
            if acceleration >= 0:
                # necessary torque is N*m = kg*m*acceleration
                torque_ratio = mass / vehicle.chassis.max_torque
                throttle = np.clip(acceleration * torque_ratio, 0, 1)
                brake = 0
            else:
                throttle = 0
                # necessary torque is N*m = kg*m*acceleration
                torque_ratio = mass / vehicle.chassis.max_btorque
                brake = np.clip(-acceleration * torque_ratio, 0, 1)
            steering = np.clip(
                dt * -angular_velocity * vehicle.chassis.steering_ratio, -1, 1
            )
            vehicle.control(throttle=throttle, brake=brake, steering=steering)

        else:
            raise Exception("unsupported chassis type")
