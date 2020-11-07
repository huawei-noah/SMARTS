import pytest
from smarts.core.coordinates import Heading, Pose
from smarts.core.scenario import Scenario
from smarts.core.smarts import SMARTS
from smarts.core.sumo_traffic_simulation import SumoTrafficSimulation
from smarts.core.tests.helpers.providers import MockProvider
import smarts.sstudio.types as t
from smarts.sstudio import gen_scenario
from helpers.scenario import temp_scenario


@pytest.fixture
def bubble():
    """
    |(93)  |(95)     (100)     (105)|  (107)|
    """
    return t.Bubble(
        zone=t.PositionalZone(pos=(100, 0), size=(10, 10)),
        margin=2,
        actor=t.SocialAgentActor(
            name="zoo-car", agent_locator="zoo.policies:keep-lane-agent-v0"
        ),
    )


@pytest.fixture
def scenarios(bubble):
    with temp_scenario(name="straight", map="maps/straight.net.xml") as scenario_root:
        gen_scenario(
            t.Scenario(traffic={}, bubbles=[bubble]), output_dir=scenario_root,
        )
        yield Scenario.variations_for_all_scenario_roots([str(scenario_root)], [])


@pytest.fixture
def mock_provider():
    return MockProvider()


@pytest.fixture
def smarts(scenarios, mock_provider):
    smarts_ = SMARTS(
        agent_interfaces={},
        traffic_sim=SumoTrafficSimulation(time_resolution=0.1, headless=False),
    )
    smarts_.add_provider(mock_provider)
    smarts_.reset(next(scenarios))
    yield smarts_
    smarts_.destroy()


def test_bubble_manager_state_change(smarts, mock_provider):
    index = smarts.vehicle_index

    vehicle_id = "vehicle-1"
    state_at_position = {
        # Outside airlock and bubble
        (92, 0, 0): (False, False),
        # Inside airlock, begin collecting experiences, but don't hijack
        (94, 0, 0): (True, False),
        # Entered bubble, now hijack
        (100, 0, 0): (False, True),
        # Leave bubble into exiting airlock
        (106, 0, 0): (False, True),
        # Exit bubble and airlock, now relinquish
        (108, 0, 0): (False, False),
    }

    for position, (shadowed, hijacked) in state_at_position.items():
        mock_provider.override_next_vehicle_state(
            vehicle_id=vehicle_id,
            pose=Pose.from_center(position, Heading(-90)),
            speed=10,
        )

        # Providers must be disjoint
        if index.vehicle_is_hijacked(vehicle_id):
            mock_provider.clear_next_vehicle_state()

            while (
                index.vehicle_is_hijacked(vehicle_id)
                and index.vehicle_position(vehicle_id)[0] < position[0]
            ):
                smarts.step({})
        else:
            smarts.step({})

        got_shadowed = index.vehicle_is_shadowed(vehicle_id)
        got_hijacked = index.vehicle_is_hijacked(vehicle_id)
        assert_msg = (
            f"position={position}\n"
            f"\t(expected: shadowed={shadowed}, hijacked={hijacked})\n"
            f"\t(received: shadowed={got_shadowed}, hijacked={got_hijacked})"
        )
        assert got_shadowed == shadowed, assert_msg
        assert got_hijacked == hijacked, assert_msg


# @mock.patch.object(Vehicle, "position")
# def test_bubble_manager_limit(road_network, bubble):
#     limit = 2
#     bubble = replace(bubble, limit=limit)
#     manager = BubbleManager([bubble], road_network)

#     vehicles_captured = [
#         Vehicle(
#             id=f"vehicle-{i}",
#             pose=Pose.from_center((0, 0, 0), Heading(0)),
#             showbase=mock.MagicMock(),
#             chassis=mock.Mock(),
#         )
#         for i in range(limit)
#     ]

#     vehicles_not_captured = [
#         Vehicle(
#             id=f"vehicle-{i}",
#             pose=Pose.from_center((0, 0, 0), Heading(0)),
#             showbase=mock.MagicMock(),
#             chassis=mock.Mock(),
#         )
#         for i in range(5)
#     ]

#     for position in [(-8, 0), (-6, 0), (-3, 0), (0, 0), (6, 0), (8, 0)]:
#         for vehicle in vehicles_captured:
#             vehicle.position = position

#         for vehicle in vehicles_not_captured:
#             vehicle.position = position

#         change = manager.step_bubble_state(vehicles_captured, vehicles_not_captured)
#         vehicle_ids_in_bubble = manager.vehicle_ids_in_bubble(bubble)
#         assert len(vehicle_ids_in_bubble) <= limit
#         assert set(vehicle_ids_in_bubble).issubset(
#             set([v.id for v in vehicles_captured])
#         )

#     manager.teardown()


# @mock.patch.object(Vehicle, "position")
# def test_vehicle_spawned_in_bubble_is_not_captured(road_network, bubble):
#     manager = BubbleManager([bubble], road_network)

#     # Spawned inside bubble, didn't "drive through" airlocking region, so should _not_
#     # get captured
#     vehicle = Vehicle(
#         id="vehicle-1",
#         pose=Pose.from_center((0, 0, 0), Heading(0)),
#         showbase=mock.MagicMock(),
#         chassis=mock.Mock(),
#     )

#     change = manager.step_bubble_state([vehicle], [])
#     assert len(manager.vehicle_ids_in_bubble(bubble)) == 0

#     # Spawned vehicle drove through airlock so _should_ get captured
#     vehicle = Vehicle(
#         id="vehicle-2",
#         pose=Pose.from_center((-8, 0, 0), Heading(0)),
#         showbase=mock.MagicMock(),
#         chassis=mock.Mock(),
#     )

#     change = manager.step_bubble_state([vehicle], [])
#     vehicle.position = (-6, 0)

#     change = manager.step_bubble_state([vehicle], [])
#     assert len(manager.vehicle_ids_in_bubble(bubble)) == 1

#     manager.teardown()
