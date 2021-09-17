# MIT License
#
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
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

# A basic implementation of the "Intelligent Driver Model". See
# https://en.wikipedia.org/wiki/Intelligent_driver_model for more details.

import math
import numpy as np
import ultra.adapters as adapters

from smarts.core.agent import Agent, AgentSpec
from smarts.core.agent_interface import AgentInterface, NeighborhoodVehicles, Waypoints
from smarts.core.controllers import ActionSpaceType
from typing import Sequence, Tuple

TIMESTEP_SEC = 0.1  # Environment timestep.
VEHICLE_LENGTH = 3.6
MAX_EPISODE_STEPS = 500


def distance_between_points(
    p1_x: float, p1_y: float, p2_x: float, p2_y: float
) -> float:
    return math.sqrt((p2_x - p1_x) ** 2 + (p2_y - p1_y) ** 2)


def calculate_curvature(waypoints: Sequence[Tuple[float, float]]) -> float:
    """Fit a quadratic curve to the waypoints and obtain the coefficients of the
    polynomial. Then, calculate the curvature, R, using the formula:
        R = |y''(x)| / (1 + (y'(x))^2)^1.5
    """
    # Offset each waypoint by a small amount to avoid invalid calculations.
    xs = np.array([waypoint[0] + i * 1e-6 for i, waypoint in enumerate(waypoints)])
    ys = np.array([waypoint[1] + i * 1e-6 for i, waypoint in enumerate(waypoints)])

    coeffs, _, _, _, _ = np.polyfit(xs, ys, deg=2, full=True)

    y_prime = 2 * coeffs[0] * xs[0] + coeffs[1]
    y_prime_prime = 2 * coeffs[0]
    curvature = abs(y_prime_prime) / (1 + y_prime ** 2) ** 1.5

    return curvature


class IDMAgent(Agent):
    STEERING_FACTOR = 6

    def __init__(
        self,
        v0: float = 7.0,
        s0: float = 2.5,
        T: float = 1.5,
        a: float = 3.0,
        b: float = 4.0,
        delta: float = 4.0,
    ):
        self._v0 = v0  # The desired velocity.
        self._s0 = s0  # A minimum desired net distance from the car ahead.
        self._T = T  # The minimum possible time to the vehicle in front.
        self._a = a  # The maximum vehicle acceleration.
        self._b = b  # Comfortable braking deceleration.
        self._delta = delta  # An exponent, usually set to 4.

    def act(self, observation) -> Sequence[float]:
        # Get all vehicles that are in front of this ego vehicle, and that are facing
        # approximately the same direction (within +/- pi / 4 of the ego's heading).
        vehicles_to_look_at = [
            vehicle
            for vehicle in observation["social_vehicles"]
            if (
                not all(attribute == 0.0 for attribute in vehicle)
                and vehicle[1] > 0.0
                and abs(vehicle[2]) <= (math.pi / 4)
            )
        ]

        # Obtain the unnormalized relative x, relative y, and speed of the ego vehicle.
        this_x = 0.0
        this_y = 0.0
        this_speed = observation["low_dim_states"][0] * 30.0

        if len(vehicles_to_look_at) > 0:
            # Adjust speed based on the nearest vehicle that is in front of the ego
            # vehicle and facing in roughly the same direction.
            other = vehicles_to_look_at[0]

            # Obtain the unnormalized relative x, relative y, and speed of the nearest
            # social vehicle.
            other_x = other[0] * 100.0
            other_y = other[1] * 100.0
            other_speed = other[2] * 30.0

            # Find the distance between the ego vehicle and the nearest social vehicle.
            distance_between = distance_between_points(this_x, this_y, other_x, other_y)

            # Calculate the desired speed based on the IDM equations.
            s_alpha = distance_between - VEHICLE_LENGTH
            s_star = (
                self._s0
                + this_speed * self._T
                + (this_speed * (this_speed - other_speed))
                / (2 * math.sqrt(self._a * self._b))
            )
            acceleration = self._a * (
                1 - (this_speed / self._v0) ** self._delta - (s_star / s_alpha) ** 2
            )
            speed = this_speed + acceleration * TIMESTEP_SEC
        else:
            # Set speed baesd on the target speed.
            speed = self._v0

        waypoints = [
            (
                observation["low_dim_states"][2 * i],  # Waypoint's relative x.
                observation["low_dim_states"][2 * i + 1],  # Waypoint's relative y.
            )
            for i in range(3, 13)  # Get the nearest 10 waypoints.
        ]
        curvature = calculate_curvature(waypoints)  # Obtain a sense of curvature.

        # Adjust speed based on the curvature. The greater the curvature, the less
        # the car's speed should be. Values have been set by observing runs on different
        # types of scenarios.
        if curvature < 0.1:
            pass
        elif curvature < 0.3:
            speed = 5.0
        elif curvature < 0.6:
            speed = 3.0
        elif curvature < 1.0:
            speed = 1.0
        else:
            speed = 0.5

        # Calculate the throttle and brake based on the desired speed.
        if speed > this_speed:
            throttle = (speed - this_speed) / this_speed
            brake = 0.0
        else:
            throttle = 0.0
            brake = -(speed - this_speed) / this_speed

        # Make the steering angle of the wheel proportional to the angle error.
        steering = -IDMAgent.STEERING_FACTOR * observation["low_dim_states"][3]

        return [throttle, brake, steering]


agent_spec = AgentSpec(
    interface=AgentInterface(
        max_episode_steps=MAX_EPISODE_STEPS,
        neighborhood_vehicles=NeighborhoodVehicles(radius=200.0),
        waypoints=Waypoints(lookahead=20),
        action=ActionSpaceType.Continuous,
    ),
    agent_builder=IDMAgent,
    agent_params={},
    observation_adapter=adapters.default_observation_vector_adapter.adapt,
    info_adapter=adapters.default_info_adapter.adapt,
)
