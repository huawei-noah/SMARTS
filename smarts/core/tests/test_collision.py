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
import time
from pathlib import Path

import pytest
from helpers.scenario import temp_scenario

from smarts.core import models
from smarts.core.agent_interface import ActionSpaceType, AgentInterface
from smarts.core.chassis import AckermannChassis, BoxChassis
from smarts.core.coordinates import Heading, Pose
from smarts.core.scenario import Scenario
from smarts.core.smarts import SMARTS
from smarts.core.sumo_traffic_simulation import SumoTrafficSimulation
from smarts.core.utils import pybullet
from smarts.core.utils.pybullet import bullet_client as bc
from smarts.core.vehicle import VEHICLE_CONFIGS
from smarts.sstudio import gen_scenario
from smarts.sstudio import types as t


@pytest.fixture
def bullet_client():
    client = bc.BulletClient(pybullet.DIRECT)
    client.setGravity(0, 0, -9.8)

    path = Path(__file__).parent / "../smarts/core/models/plane.urdf"
    with pkg_resources.path(models, "plane.urdf") as path:
        plane_path = str(path.absolute())
    client.loadURDF(plane_path, useFixedBase=True)

    yield client
    client.disconnect()


def step_with_vehicle_commands(
    bv: AckermannChassis, steps, throttle=0, brake=0, steering=0
):
    collisions = []
    for _ in range(steps):
        bv.control(throttle, brake, steering)
        bv._client.stepSimulation()
        collisions.extend(bv.contact_points)
    return collisions


def test_collision(bullet_client: bc.BulletClient):
    """Spawn overlap to check for the most basic collision"""

    chassis = AckermannChassis(
        Pose.from_center([0, 0, 0], Heading(-math.pi * 0.5)), bullet_client
    )

    b_chassis = BoxChassis(
        Pose.from_center([0, 0, 0], Heading(0)),
        speed=0,
        dimensions=VEHICLE_CONFIGS["passenger"].dimensions,
        bullet_client=chassis._client,
    )

    collisions = step_with_vehicle_commands(chassis, steps=1)
    assert b_chassis.bullet_id in [c.bullet_id for c in collisions]
    assert len(collisions) > 0


def test_non_collision(bullet_client: bc.BulletClient):
    """Spawn without overlap to check for the most basic collision"""

    GROUND_ID = 0
    chassis = AckermannChassis(
        Pose.from_center([0, 0, 0], Heading(-math.pi * 0.5)), bullet_client
    )

    b_chassis = BoxChassis(
        Pose.from_center([0, 10, 0], Heading(0)),
        speed=0,
        dimensions=VEHICLE_CONFIGS["passenger"].dimensions,
        bullet_client=chassis._client,
    )

    collisions = step_with_vehicle_commands(chassis, steps=1)
    collided_bullet_ids = [c.bullet_id for c in collisions]
    assert b_chassis.bullet_id not in collided_bullet_ids
    assert (
        len(collisions) == 0
        or len(collided_bullet_ids) == 1
        and GROUND_ID in set(collided_bullet_ids)
    )


def test_collision_collide_with_standing_vehicle(bullet_client: bc.BulletClient):
    """Run a vehicle at a standing vehicle as fast as possible."""
    chassis = AckermannChassis(
        Pose.from_center([10, 0, 0], Heading(math.pi * 0.5)), bullet_client
    )

    b_chassis = BoxChassis(
        Pose.from_center([0, 0, 0], Heading(0)),
        speed=0,
        dimensions=VEHICLE_CONFIGS["passenger"].dimensions,
        bullet_client=chassis._client,
    )
    collisions = step_with_vehicle_commands(chassis, steps=1000, throttle=1, steering=0)
    assert len(collisions) > 0
    assert b_chassis.bullet_id in [c.bullet_id for c in collisions]


def _joust(wkc: AckermannChassis, bkc: AckermannChassis, steps, throttle=1):
    collisions = ([], [])
    for _ in range(steps):
        wkc.control(throttle)
        bkc.control(throttle)
        bkc._client.stepSimulation()
        collisions[0].extend(wkc.contact_points)
        collisions[1].extend(bkc.contact_points)
    return collisions


def test_collision_joust(bullet_client: bc.BulletClient):
    """ Run two agents at each other to test for clipping. """
    GROUND_ID = 0
    white_knight_chassis = AckermannChassis(
        Pose.from_center([10, 0, 0], Heading(math.pi * 0.5)), bullet_client
    )
    black_knight_chassis = AckermannChassis(
        Pose.from_center([-10, 0, 0], Heading(-math.pi * 0.5)), bullet_client
    )

    wkc_collisions, bkc_collisions = _joust(
        white_knight_chassis, black_knight_chassis, steps=10000
    )

    assert len(wkc_collisions) > 0
    assert len(bkc_collisions) > 0

    assert white_knight_chassis.bullet_id in [c.bullet_id for c in bkc_collisions]
    assert black_knight_chassis.bullet_id in [c.bullet_id for c in wkc_collisions]


def test_ackerman_chassis_size_unchanged(bullet_client: bc.BulletClient):
    """Test that the ackerman chassis size has not changed accidentally by packing it around itself
    with no forces and then check for collisions after a few steps."""
    bullet_client.setGravity(0, 0, 0)
    separation_for_collision_error = 0.05
    original_vehicle_dimensions = VEHICLE_CONFIGS["passenger"].dimensions

    shared_heading = Heading(0)
    chassis = AckermannChassis(
        Pose.from_center([0, 0, 0], shared_heading), bullet_client
    )
    chassis_n = BoxChassis(
        Pose.from_center(
            [
                0,
                (original_vehicle_dimensions.length + separation_for_collision_error),
                0,
            ],
            shared_heading,
        ),
        speed=0,
        dimensions=VEHICLE_CONFIGS["passenger"].dimensions,
        bullet_client=bullet_client,
    )
    chassis_e = BoxChassis(
        Pose.from_center(
            [
                (original_vehicle_dimensions.width + separation_for_collision_error),
                0,
                0,
            ],
            shared_heading,
        ),
        speed=0,
        dimensions=VEHICLE_CONFIGS["passenger"].dimensions,
        bullet_client=bullet_client,
    )
    chassis_s = BoxChassis(
        Pose.from_center(
            [
                0,
                -(original_vehicle_dimensions.length + separation_for_collision_error),
                0,
            ],
            shared_heading,
        ),
        speed=0,
        dimensions=VEHICLE_CONFIGS["passenger"].dimensions,
        bullet_client=bullet_client,
    )
    chassis_w = BoxChassis(
        Pose.from_center(
            [
                -(original_vehicle_dimensions.width + separation_for_collision_error),
                0,
                0,
            ],
            shared_heading,
        ),
        speed=0,
        dimensions=VEHICLE_CONFIGS["passenger"].dimensions,
        bullet_client=bullet_client,
    )
    collisions = step_with_vehicle_commands(chassis, steps=10)
    assert len(collisions) < 1


AGENT_1 = "Agent_007"
AGENT_2 = "Agent_008"


@pytest.fixture
def scenarios():
    with temp_scenario(name="straight", map="maps/straight.net.xml") as scenario_root:
        ego_missions = [
            # missions of laner and buddha
            t.Mission(
                t.Route(
                    begin=("west", 0, 30),
                    end=("east", 0, "max"),
                )
            ),
            t.Mission(
                t.Route(
                    begin=("west", 0, 40),
                    end=("east", 0, "max"),
                )
            ),
        ]
        gen_scenario(
            t.Scenario(ego_missions=ego_missions),
            output_dir=scenario_root,
        )

        yield Scenario.variations_for_all_scenario_roots(
            [str(scenario_root)], [AGENT_1, AGENT_2]
        )


@pytest.fixture
def smarts():
    laner = AgentInterface(
        max_episode_steps=1000,
        action=ActionSpaceType.Lane,
    )
    buddha = AgentInterface(
        max_episode_steps=1000,
        action=ActionSpaceType.Lane,
    )
    agents = {AGENT_1: laner, AGENT_2: buddha}
    smarts = SMARTS(
        agents,
        traffic_sim=SumoTrafficSimulation(headless=True),
        envision=None,
    )

    yield smarts
    smarts.destroy()


def test_sim_level_collision(smarts, scenarios):
    scenario = next(scenarios)
    smarts.reset(scenario)

    collisions = []
    agent_dones = []

    for _ in range(30):
        observations, _, dones, _ = smarts.step({AGENT_1: "keep_lane"})
        if AGENT_1 in observations:
            collisions.extend(observations[AGENT_1].events.collisions)
            agent_dones.append(dones[AGENT_1])

    assert len(collisions) > 0
    assert any(agent_dones)
