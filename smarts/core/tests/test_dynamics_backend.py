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
import math

import numpy as np
import pytest

from smarts.core.chassis import AckermannChassis
from smarts.core.coordinates import Heading, Pose
from smarts.core.utils import pybullet
from smarts.core.utils.pybullet import bullet_client as bc


@pytest.fixture
def bullet_client():
    client = bc.BulletClient(pybullet.DIRECT)
    yield client
    client.disconnect()


@pytest.fixture
def chassis(bullet_client):
    return AckermannChassis(
        Pose.from_center([0, 0, 0], Heading(math.pi * 0.5)), bullet_client
    )


def step_with_vehicle_commands(bv, steps, throttle=0, brake=0, steering=0):
    for _ in range(steps):
        bv.control(throttle, brake, steering)
        bv._client.stepSimulation()


def test_steering_direction(chassis):
    step_with_vehicle_commands(chassis, steps=100, steering=0)
    assert math.isclose(chassis.steering, 0, rel_tol=1e-2)

    # steer as far right as we can and test that the steering values we read
    # back also correspond to a right turn.
    step_with_vehicle_commands(chassis, steps=100, steering=1)
    assert chassis.steering > 0

    # steer as far left as we can and test that the steering values we read
    # back also correspond to a left turn.
    step_with_vehicle_commands(chassis, steps=100, steering=-1)
    assert chassis.steering < 0


def test_set_pose(chassis):
    chassis.set_pose(Pose.from_center([137, -5.8, 1], Heading(1.8)))
    position, heading = chassis.pose.position, chassis.pose.heading

    assert np.linalg.norm(position - np.array([137, -5.8, 1])) < 1e-16
    assert np.isclose(heading, 1.8)
