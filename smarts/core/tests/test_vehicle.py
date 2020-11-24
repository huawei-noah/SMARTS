import math

import numpy as np
import pytest
from direct.showbase.ShowBase import ShowBase

from smarts.core.utils import pybullet
from smarts.core.utils.pybullet import bullet_client as bc
from smarts.core.coordinates import BoundingBox, Heading, Pose
from smarts.core.vehicle import VEHICLE_CONFIGS, Vehicle, VehicleState
from smarts.core.chassis import BoxChassis


@pytest.fixture
def showbase():
    showbase = ShowBase()
    yield showbase
    showbase.destroy()


@pytest.fixture
def bullet_client():
    client = bc.BulletClient(pybullet.DIRECT)
    yield client
    client.disconnect()


# TODO: Clean up these tests and fixtures
@pytest.fixture
def position():
    return [-80, -80, 5]


@pytest.fixture
def heading():
    return Heading.from_panda3d(-250)


@pytest.fixture
def speed():
    return 50


@pytest.fixture
def social_vehicle(position, heading, speed, showbase, bullet_client):
    pose = Pose.from_center(position, heading)
    chassis = BoxChassis(
        pose=pose,
        speed=speed,
        dimensions=VEHICLE_CONFIGS["passenger"].dimensions,
        bullet_client=bullet_client,
    )
    return Vehicle(id="sv-132", pose=pose, showbase=showbase, chassis=chassis)


@pytest.fixture
def provider_vehicle(position, heading, speed):
    return VehicleState(
        vehicle_id="sv-132",
        vehicle_type="truck",
        pose=Pose.from_center(position, heading),
        dimensions=BoundingBox(length=3, width=1, height=2),
        speed=speed,
        source="TESTS",
    )


def test_update_from_traffic_sim(social_vehicle, provider_vehicle):
    social_vehicle.control(
        pose=provider_vehicle.pose, speed=provider_vehicle.speed,
    )

    sv_position, sv_heading = social_vehicle.pose.as_sumo(
        social_vehicle.length, Heading(0)
    )
    provider_position, provider_heading = provider_vehicle.pose.as_sumo(
        social_vehicle.length, Heading(0)
    )
    assert np.isclose(sv_position, provider_position, rtol=1e-02).all()

    assert math.isclose(sv_heading, provider_heading, rel_tol=1e-05,)
    assert social_vehicle.speed == provider_vehicle.speed


def test_create_social_vehicle(showbase, bullet_client):
    chassis = BoxChassis(
        pose=Pose.from_center((0, 0, 0), Heading(0)),
        speed=0,
        dimensions=BoundingBox(length=3, width=1, height=1),
        bullet_client=bullet_client,
    )

    car = Vehicle(
        id="sv-132",
        pose=Pose.from_center((0, 0, 0), Heading(0)),
        showbase=showbase,
        chassis=chassis,
        sumo_vehicle_type="passenger",
    )
    assert car.vehicle_type == "car"

    truck = Vehicle(
        id="sv-132",
        pose=Pose.from_center((0, 0, 0), Heading(0)),
        showbase=showbase,
        chassis=chassis,
        sumo_vehicle_type="truck",
    )
    assert truck.vehicle_type == "truck"


def test_vehicle_bounding_box(showbase, bullet_client):
    pose = Pose.from_center((1, 1, 0), Heading(0))
    chassis = BoxChassis(
        pose=pose,
        speed=0,
        dimensions=BoundingBox(length=3, width=1, height=1),
        bullet_client=bullet_client,
    )

    vehicle = Vehicle(
        id="vehicle-0",
        pose=pose,
        showbase=showbase,
        chassis=chassis,
        sumo_vehicle_type="passenger",
    )
    for coordinates in zip(
        vehicle.bounding_box, [[0.5, 2.5], (1.5, 2.5), (1.5, -0.5), (0.5, -0.5)]
    ):
        assert np.array_equal(coordinates[0], coordinates[1])
