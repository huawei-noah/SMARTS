import math

from unittest import mock
import numpy as np
import pytest

import random
from pathlib import Path

from smarts.core.utils import pybullet
from smarts.core.utils.pybullet import bullet_client as bc
from smarts.core.chassis import AckermannChassis
from smarts.core.controllers import (
    TrajectoryTrackingController,
    TrajectoryTrackingControllerState,
)
from smarts.core.coordinates import Heading, Pose
from smarts.core.vehicle import Vehicle


time_step = 0.1


@pytest.fixture
def bullet_client(TIMESTEP_SEC=time_step):
    client = bc.BulletClient(pybullet.DIRECT)
    client.resetSimulation()
    client.setGravity(0, 0, -9.8)
    client.setPhysicsEngineParameter(
        fixedTimeStep=TIMESTEP_SEC, numSubSteps=int(TIMESTEP_SEC / (1 / 240)),
    )
    path = Path(__file__).parent / "../models/plane.urdf"
    path = str(path.absolute())
    plane_body_id = client.loadURDF(path, useFixedBase=True)
    yield client
    client.disconnect()


@pytest.fixture
def vehicle(bullet_client, TIMESTEP_SEC=time_step):
    pose = Pose.from_center((0, 0, 0), Heading(0))
    vehicle1 = Vehicle(
        id="vehicle",
        pose=pose,
        showbase=mock.MagicMock(),
        chassis=AckermannChassis(pose=pose, bullet_client=bullet_client,),
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
def build_trajectory(radius, omega, step_num, TIMESTEP_SEC=time_step):
    num_trajectory_points = 15
    R = radius
    omega_1 = omega
    omega_2 = omega
    if step_num > 3.14 / (TIMESTEP_SEC * omega_1):
        Omega = omega_2
        alph = ((omega_1 - omega_2) / omega_2) * 3.14 / (TIMESTEP_SEC * omega_1)
    else:
        Omega = omega_1
        alph = 0

    trajectory = [
        [
            -(R - R * math.cos((step_num + i + alph) * Omega * TIMESTEP_SEC))
            for i in range(num_trajectory_points)
        ],
        [
            R * math.sin((step_num + i + alph) * Omega * TIMESTEP_SEC)
            for i in range(num_trajectory_points)
        ],
        [
            (step_num + i + alph) * Omega * TIMESTEP_SEC
            for i in range(num_trajectory_points)
        ],
        [R * Omega for i in range(num_trajectory_points)],
    ]
    return trajectory


def step_with_vehicle_commands(
    bullet_client, vehicle, radius, omega, TIMESTEP_SEC=time_step
):
    prev_friction_sum = None
    # Proceed till the end of half of the circle.
    n_steps = int(0.5 * 3.14 / (omega * TIMESTEP_SEC))

    desired_trajectory = []
    controller_state = TrajectoryTrackingControllerState()
    for step_num in range(n_steps):
        desired_trajectory = build_trajectory(radius, omega, step_num)

        TrajectoryTrackingController.perform_trajectory_tracking_PD(
            desired_trajectory, vehicle, controller_state, dt_sec=TIMESTEP_SEC,
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
