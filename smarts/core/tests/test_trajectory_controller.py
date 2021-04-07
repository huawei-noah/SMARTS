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
import importlib.resources as pkg_resources
import math
import os
import random
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import yaml

from smarts.core import models
from smarts.core.chassis import AckermannChassis
from smarts.core.controllers import (
    TrajectoryTrackingController,
    TrajectoryTrackingControllerState,
)
from smarts.core.coordinates import Heading, Pose
from smarts.core.utils import pybullet
from smarts.core.utils.pybullet import bullet_client as bc
from smarts.core.vehicle import Vehicle

time_step = 0.1


@pytest.fixture(params=["sedan", "bus"])
def vehicle_controller_file(request):
    vehicle_file_name = request.param + ".urdf"
    if request.param == "sedan":
        vehicle_file_name = "vehicle.urdf"

    with pkg_resources.path(models, vehicle_file_name) as path:
        vehicle_file_path = str(path.absolute())
    with pkg_resources.path(models, "controller_parameters.yaml") as controller_path:
        controller_filepath = str(controller_path.absolute())
    with open(controller_filepath, "r") as controller_file:
        vehicle_controller_file_path = yaml.safe_load(controller_file)[request.param]

    return (vehicle_file_path, vehicle_controller_file_path)


@pytest.fixture
def bullet_client(timestep_sec=time_step):
    client = bc.BulletClient(pybullet.DIRECT)
    client.resetSimulation()
    client.setGravity(0, 0, -9.8)
    client.setPhysicsEngineParameter(
        fixedTimeStep=timestep_sec,
        numSubSteps=int(timestep_sec / (1 / 240)),
    )
    path = Path(__file__).parent / "../models/plane.urdf"
    path = str(path.absolute())
    plane_body_id = client.loadURDF(path, useFixedBase=True)
    yield client
    client.disconnect()


@pytest.fixture
def vehicle(bullet_client, vehicle_controller_file, timestep_sec=time_step):
    pose = Pose.from_center((0, 0, 0), Heading(0))
    vehicle1 = Vehicle(
        id="vehicle",
        pose=pose,
        chassis=AckermannChassis(
            pose=pose,
            bullet_client=bullet_client,
            vehicle_filepath=vehicle_controller_file[0],
            controller_parameters=vehicle_controller_file[1],
        ),
    )
    return vehicle1


# We test 3 values for radius of the circular trajectory
@pytest.fixture(params=np.arange(10, 30, 10))
def radius(request):
    return request.param


# We test 3 values for vehicle yaw rate of the circular trajectory
@pytest.fixture(params=np.arange(0.1, 0.21, 0.05))
def omega(request):
    return request.param


# We use circular trajectory with different radius and yaw rate
def build_trajectory(radius, omega, step_num, timestep_sec=time_step):
    num_trajectory_points = 15
    R = radius
    omega_1 = omega
    omega_2 = omega
    if step_num > 3.14 / (timestep_sec * omega_1):
        Omega = omega_2
        alph = ((omega_1 - omega_2) / omega_2) * 3.14 / (timestep_sec * omega_1)
    else:
        Omega = omega_1
        alph = 0
    desheadi = step_num * Omega * timestep_sec
    trajectory = [
        [
            -(R - R * math.cos((step_num + i + alph) * Omega * timestep_sec))
            for i in range(num_trajectory_points)
        ],
        [
            R * math.sin((step_num + i + alph) * Omega * timestep_sec)
            for i in range(num_trajectory_points)
        ],
        [
            (step_num + i + alph) * Omega * timestep_sec
            for i in range(num_trajectory_points)
        ],
        [R * Omega for i in range(num_trajectory_points)],
    ]
    return trajectory


def step_with_vehicle_commands(
    bullet_client, vehicle, radius, omega, timestep_sec=time_step
):
    prev_friction_sum = None
    # Proceed till the end of half of the circle.
    n_steps = int(0.5 * 3.14 / (omega * timestep_sec))

    desired_trajectory = []
    controller_state = TrajectoryTrackingControllerState()
    for step_num in range(n_steps):
        desired_trajectory = build_trajectory(radius, omega, step_num)

        TrajectoryTrackingController.perform_trajectory_tracking_PD(
            desired_trajectory,
            vehicle,
            controller_state,
            dt_sec=timestep_sec,
        )

        bullet_client.stepSimulation()

    final_error = math.sqrt(
        (vehicle.position[0] - desired_trajectory[0][0]) ** 2
        + (vehicle.position[1] - desired_trajectory[1][0]) ** 2
    )
    return final_error


def test_trajectory_tracking(bullet_client, vehicle, radius, omega):
    final_error = step_with_vehicle_commands(bullet_client, vehicle, radius, omega)
    assert final_error <= 10
