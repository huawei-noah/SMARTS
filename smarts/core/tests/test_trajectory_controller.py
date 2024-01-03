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
# FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
import importlib.resources as pkg_resources
import math

import numpy as np
import pytest

import smarts.assets.vehicles.controller_params
import smarts.assets.vehicles.dynamics_model
from smarts.core.chassis import AckermannChassis
from smarts.core.controllers import (
    TrajectoryTrackingController,
    TrajectoryTrackingControllerState,
)
from smarts.core.coordinates import Heading, Pose
from smarts.core.utils import pybullet
from smarts.core.utils.pybullet import SafeBulletClient
from smarts.core.utils.resources import (
    VehicleDefinitions,
    load_vehicle_definitions_list,
    load_yaml_config_with_substitution,
)
from smarts.core.vehicle import Vehicle

time_step = 0.1


@pytest.fixture
def vehicle_definitions_list() -> VehicleDefinitions:
    return load_vehicle_definitions_list(None)


@pytest.fixture(params=["bus", "sedan", "truck"])
def vehicle_definition(
    vehicle_definitions_list: VehicleDefinitions, request: pytest.FixtureRequest
):
    return vehicle_definitions_list.load_vehicle_definition(request.param)


@pytest.fixture
def bullet_client(fixed_timestep_sec=time_step):
    client = SafeBulletClient(pybullet.DIRECT)
    client.resetSimulation()
    client.setGravity(0, 0, -9.8)
    client.setPhysicsEngineParameter(
        fixedTimeStep=fixed_timestep_sec,
        numSubSteps=int(fixed_timestep_sec / (1 / 240)),
    )
    with pkg_resources.path(smarts.assets, "plane.urdf") as path:
        path = str(path.absolute())
        plane_body_id = client.loadURDF(path, useFixedBase=True)
    yield client
    client.disconnect()


@pytest.fixture
def vehicle(bullet_client, vehicle_definition, fixed_timestep_sec=time_step):
    pose = Pose.from_center((0, 0, 0), Heading(0))
    vehicle1 = Vehicle(
        id="vehicle",
        chassis=AckermannChassis(
            pose=pose,
            bullet_client=bullet_client,
            vehicle_dynamics_filepath=vehicle_definition["dynamics_model"],
            controller_parameters=load_yaml_config_with_substitution(
                vehicle_definition["controller_params"]
            ),
            chassis_parameters=load_yaml_config_with_substitution(
                vehicle_definition["chassis_params"]
            ),
        ),
        visual_model_filepath=None,
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
def build_trajectory(radius, omega, step_num, fixed_timestep_sec=time_step):
    num_trajectory_points = 15
    R = radius
    omega_1 = omega
    omega_2 = omega
    if step_num > 3.14 / (fixed_timestep_sec * omega_1):
        Omega = omega_2
        alph = ((omega_1 - omega_2) / omega_2) * 3.14 / (fixed_timestep_sec * omega_1)
    else:
        Omega = omega_1
        alph = 0
    desheadi = step_num * Omega * fixed_timestep_sec
    trajectory = [
        [
            -(R - R * math.cos((step_num + i + alph) * Omega * fixed_timestep_sec))
            for i in range(num_trajectory_points)
        ],
        [
            R * math.sin((step_num + i + alph) * Omega * fixed_timestep_sec)
            for i in range(num_trajectory_points)
        ],
        [
            (step_num + i + alph) * Omega * fixed_timestep_sec
            for i in range(num_trajectory_points)
        ],
        [R * Omega for i in range(num_trajectory_points)],
    ]
    return trajectory


def step_with_vehicle_commands(
    bullet_client, vehicle, radius, omega, fixed_timestep_sec=time_step
):
    prev_friction_sum = None
    # Proceed till the end of half of the circle.
    n_steps = int(0.5 * 3.14 / (omega * fixed_timestep_sec))

    desired_trajectory = []
    controller_state = TrajectoryTrackingControllerState()
    for step_num in range(n_steps):
        desired_trajectory = build_trajectory(radius, omega, step_num)

        TrajectoryTrackingController.perform_trajectory_tracking_PD(
            desired_trajectory,
            vehicle,
            controller_state,
            dt_sec=fixed_timestep_sec,
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
