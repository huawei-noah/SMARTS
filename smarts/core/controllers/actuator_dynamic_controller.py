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


class ActuatorDynamicControllerState:
    def __init__(self):
        self.last_steering_angle = 0


class ActuatorDynamicController:
    @classmethod
    def perform_action(cls, vehicle, action, state, dt_sec):
        throttle, brake, steering_change = action

        # The following is the normalized steering change
        # under the assumption that the steering angles at
        # wheels can reach to their maximum values in one
        # second.
        clipped_steering_change = np.clip(steering_change, -1, 1,)

        p = 0.001  # XXX: Theorized to improve stability, this has yet to be seen.
        steering = np.clip(
            (1 - p) * state.last_steering_angle + clipped_steering_change * dt_sec,
            -1,
            1,
        )

        vehicle.control(
            throttle=np.clip(throttle, 0.0, 1.0),
            brake=np.clip(brake, 0.0, 1.0),
            steering=steering,
        )

        state.last_steering_angle = steering
