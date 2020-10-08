import math
from enum import Enum
from functools import partial

from numpy.linalg import matrix_power
import numpy as np

from scipy import signal
from cvxopt import matrix, solvers
from qpsolvers import solve_qp

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
