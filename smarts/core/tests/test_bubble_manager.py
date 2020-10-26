from unittest import mock
from dataclasses import replace

import pytest

from smarts.core.coordinates import Pose, Heading
from smarts.core.vehicle import Vehicle
from smarts.core.bubble_manager import BubbleManager
from smarts.core.sumo_road_network import SumoRoadNetwork
from smarts.sstudio.types import Bubble, PositionalZone, SocialAgentActor


@pytest.fixture
def bubble():
    """
    |(-7)  |(-5)     (0)     (5)|  (7)|
    """
    return Bubble(
        zone=PositionalZone(pos=(0, 0), size=(10, 10)),
        margin=2,
        actor=SocialAgentActor(name="zoo-car", agent_locator="come.find.me"),
    )


@pytest.fixture
def road_network():
    return SumoRoadNetwork.from_file("scenarios/intersections/4lane_t/map.net.xml")


@mock.patch.object(Vehicle, "position")
def test_bubble_manager_state_change(vehicle, bubble):
    manager = BubbleManager([bubble], road_network)
    sim = mock.Mock()

    # Outside airlock and bubble
    vehicle = Vehicle(
        id="vehicle-1",
        pose=Pose.from_center((0, 0, 0), Heading(0)),
        showbase=mock.MagicMock(),
        chassis=mock.Mock(),
    )

    vehicle.position = (-8, 0)
    change = manager.step_bubble_state(sim, [vehicle], [])
    assert len(change.entered_airlock_1) == len(change.entered_bubble) == 0

    # Inside airlock, begin collecting experiences, but don't hijack
    vehicle.position = (-6, 0)
    change = manager.step_bubble_state(sim, [vehicle], [])
    assert len(change.entered_airlock_1) == 1 and len(change.entered_bubble) == 0

    # Entered bubble, now hijack
    vehicle.position = (-3, 0)
    change = manager.step_bubble_state(sim, [vehicle], [])
    assert len(change.entered_airlock_1) == 0 and len(change.entered_bubble) == 1
    assert change.entered_bubble[0][0] == vehicle.id

    # Leave bubble into exiting airlock
    vehicle.position = (6, 0)
    change = manager.step_bubble_state(sim, [], [vehicle])
    assert len(change.entered_bubble) == 0 and len(change.exited_bubble) == 1

    # Exit bubble and airlock, now relinquish
    vehicle.position = (8, 0)
    change = manager.step_bubble_state(sim, [vehicle], [])
    assert len(change.exited_bubble) == 0 and len(change.exited_airlock_2) == 1

    manager.teardown()


@mock.patch.object(Vehicle, "position")
def test_bubble_manager_limit(vehicle, bubble):
    limit = 2
    bubble = replace(bubble, limit=limit)
    manager = BubbleManager([bubble], road_network)
    sim = mock.Mock()

    vehicles_captured = [
        Vehicle(
            id=f"vehicle-{i}",
            pose=Pose.from_center((0, 0, 0), Heading(0)),
            showbase=mock.MagicMock(),
            chassis=mock.Mock(),
        )
        for i in range(limit)
    ]

    vehicles_not_captured = [
        Vehicle(
            id=f"vehicle-{i}",
            pose=Pose.from_center((0, 0, 0), Heading(0)),
            showbase=mock.MagicMock(),
            chassis=mock.Mock(),
        )
        for i in range(5)
    ]

    for position in [(-8, 0), (-6, 0), (-3, 0), (0, 0), (6, 0), (8, 0)]:
        for vehicle in vehicles_captured:
            vehicle.position = position

        for vehicle in vehicles_not_captured:
            vehicle.position = position

        change = manager.step_bubble_state(
            sim, vehicles_captured, vehicles_not_captured
        )
        vehicle_ids_in_bubble = manager.vehicle_ids_in_bubble(bubble)
        assert len(vehicle_ids_in_bubble) <= limit
        assert set(vehicle_ids_in_bubble).issubset(
            set([v.id for v in vehicles_captured])
        )

    manager.teardown()


@mock.patch.object(Vehicle, "position")
def test_vehicle_spawned_in_bubble_is_not_captured(vehicle, bubble):
    manager = BubbleManager([bubble], road_network)
    sim = mock.Mock()

    # Spawned inside bubble, didn't "drive through" airlocking region, so should _not_
    # get captured
    vehicle = Vehicle(
        id="vehicle-1",
        pose=Pose.from_center((0, 0, 0), Heading(0)),
        showbase=mock.MagicMock(),
        chassis=mock.Mock(),
    )

    change = manager.step_bubble_state(sim, [vehicle], [])
    assert len(manager.vehicle_ids_in_bubble(bubble)) == 0

    # Spawned vehicle drove through airlock so _should_ get captured
    vehicle = Vehicle(
        id="vehicle-2",
        pose=Pose.from_center((-8, 0, 0), Heading(0)),
        showbase=mock.MagicMock(),
        chassis=mock.Mock(),
    )

    change = manager.step_bubble_state(sim, [vehicle], [])
    vehicle.position = (-6, 0)

    change = manager.step_bubble_state(sim, [vehicle], [])
    assert len(manager.vehicle_ids_in_bubble(bubble)) == 1

    manager.teardown()
