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
from typing import Sequence, Set, Dict
import math
import numpy as np

from smarts.core.controllers import ActionSpaceType
from smarts.core.provider import ProviderState
from smarts.core.vehicle import VEHICLE_CONFIGS, VehicleState
from smarts.core.coordinates import Heading, Pose


class TrajectoryWithTime:
    TIME_INDEX = 0
    X_INDEX = 1
    Y_INDEX = 2
    THETA_INDEX = 3
    VEL_INDEX = 4


class TrajectoryInterpolationProvider:
    def __init__(self):
        self._is_setup = False

    @property
    def action_spaces(self) -> Set[ActionSpaceType]:
        return {ActionSpaceType.TrajectoryWithTime}

    def setup(self, scenario) -> ProviderState:
        self._is_setup = True
        return ProviderState()

    def reset(self):
        pass

    def sync(self, provider_state):
        pass  # ... we ignore other sim state here

    def teardown(self):
        self._next_provider_state = None

    def step(self, provider_actions: Dict, dt, elapsed_sim_time) -> ProviderState:
        """[summary]

        Args:

        Returns:
            ProviderState: [description]
        """

        provider_state = ProviderState()

        for vehicle_id, traj in provider_actions.items():
            pose, speed = self.perform_trajectory_interpolation(dt, traj)

            provider_state.vehicles.append(
                VehicleState(
                    vehicle_id=vehicle_id,
                    vehicle_type="passenger",
                    pose=pose,
                    dimensions=VEHICLE_CONFIGS["passenger"].dimensions,
                    speed=speed,
                    source="TrajectoryInterpolation",
                )
            )

        return provider_state

    def create_vehicle(self, provider_vehicle: VehicleState):
        pass

    @staticmethod
    def is_legal_trajectory(trajectory: np.ndarray):
        assert (
            len(trajectory[TrajectoryWithTime.TIME_INDEX]) >= 2
        ), "Length of trajectory is less than 2!"

        assert np.isfinite(
            trajectory
        ).all(), "Has nan, positive infi or nagative infi in trajectory!"

        assert (
            np.diff(trajectory[TrajectoryWithTime.TIME_INDEX]) > 0
        ).all(), "Time of trajectory is not strictly increasing!"

    @staticmethod
    def interpolate(ms0: np.ndarray, ms1: np.ndarray, time: float) -> np.ndarray:
        """Linear Interpolate between two vehicle motion state

        Returns:
            np.ndarray: New vehicle state between vehicle motion state ms0 and ms1
        """

        start_time = ms0[TrajectoryWithTime.TIME_INDEX]
        end_time = ms1[TrajectoryWithTime.TIME_INDEX]
        ratio = math.fabs((time - start_time) / (end_time - start_time))
        assert end_time >= start_time and time >= start_time

        np_motion_state = (1 - ratio) * ms0 + ratio * ms1
        CS = (1 - ratio) * math.cos(
            ms0[TrajectoryWithTime.THETA_INDEX]
        ) + ratio * math.cos(ms1[TrajectoryWithTime.THETA_INDEX])
        SN = (1 - ratio) * math.sin(
            ms0[TrajectoryWithTime.THETA_INDEX]
        ) + ratio * math.sin(ms1[TrajectoryWithTime.THETA_INDEX])
        np_motion_state[TrajectoryWithTime.THETA_INDEX] = math.atan2(SN, CS)
        return np_motion_state

    @staticmethod
    def locate_motion_state(trajectory, time) -> np.ndarray:
        end_index = 0
        for i, t in enumerate(trajectory[TrajectoryWithTime.TIME_INDEX]):
            if t > time:
                end_index = i
                break

        assert (
            end_index > 0
        ), f"Expected relative time, {time} sec, can not be located at input with-time-trajectory"

        return trajectory[:, end_index - 1], trajectory[:, end_index]

    @staticmethod
    def perform_trajectory_interpolation(
        timestep_sec,
        trajectory: np.ndarray,
    ):
        """Move vehicle by trajectory interpolation.

        Trajectory mentioned here has 5 dimensions, which are TIME, X, Y, THETA and VEL.
        TIME indicate

        If you want vehicle stop at a specific pose,
        trajectory[TrajectoryWithTime.TIME_INDEX][0] should be set as numpy.inf

        Args:
            sim : reference of smarts instance
            vehicle : vehicle to be controlled
            trajectory (np.ndarray): trajectory with time
        """
        TrajectoryInterpolationProvider.is_legal_trajectory(trajectory)

        ms0, ms1 = TrajectoryInterpolationProvider.locate_motion_state(
            trajectory, timestep_sec
        )

        speed = 0.0
        pose = []
        if math.isinf(ms0[TrajectoryWithTime.TIME_INDEX]) or math.isinf(
            ms1[TrajectoryWithTime.TIME_INDEX]
        ):
            center_position = ms0[
                TrajectoryWithTime.X_INDEX : TrajectoryWithTime.Y_INDEX + 1
            ]
            center_heading = Heading(ms0[TrajectoryWithTime.THETA_INDEX])
            pose = Pose.from_center(center_position, center_heading)
            speed = 0.0
        else:
            ms = TrajectoryInterpolationProvider.interpolate(ms0, ms1, timestep_sec)

            center_position = ms[
                TrajectoryWithTime.X_INDEX : TrajectoryWithTime.Y_INDEX + 1
            ]
            center_heading = Heading(ms[TrajectoryWithTime.THETA_INDEX])
            pose = Pose.from_center(center_position, center_heading)
            speed = ms[TrajectoryWithTime.VEL_INDEX]
        return pose, speed
