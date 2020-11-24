import math
import importlib.resources as pkg_resources
import time

from pathlib import Path

from smarts.core.utils import pybullet
from smarts.core.utils.pybullet import bullet_client as bc
import pytest

from smarts.core import models
from smarts.core.chassis import AckermannChassis, BoxChassis
from smarts.core.coordinates import Heading, Pose
from smarts.core.vehicle import VEHICLE_CONFIGS


@pytest.fixture
def bullet_client():
    client = bc.BulletClient(pybullet.DIRECT)
    client.setGravity(0, 0, -9.8)

    path = Path(__file__).parent / "../smarts/core/models/plane.urdf"
    with pkg_resources.path(models, "plane.urdf") as path:
        plane_path = str(path.absolute())
    plane_body_id = client.loadURDF(plane_path, useFixedBase=True)

    yield client
    client.disconnect()


def step_with_vehicle_commands(
    bv: AckermannChassis, steps, throttle=0, brake=0, steering=0
):
    collisions = []
    for s in range(steps):
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
