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

import pytest
from helpers.scenario import temp_scenario

import smarts.sstudio.types as t
from smarts.core.coordinates import Heading, Pose
from smarts.core.scenario import Scenario
from smarts.core.smarts import SMARTS
from smarts.core.sumo_traffic_simulation import SumoTrafficSimulation
from smarts.core.tests.helpers.providers import MockProvider
from smarts.sstudio import gen_scenario

# TODO: Add test for travelling bubbles


@pytest.fixture
def time_resolution(request):
    tr = getattr(request, "param", 0.1)
    assert tr >= 1e-10, "Should be a non-negative non-zero real number"
    return tr


@pytest.fixture
def bubble(request):
    """
    |(93)  |(95)     (100)     (105)|  (107)|
    """
    return t.Bubble(
        zone=t.PositionalZone(pos=(100, 0), size=(10, 10)),
        margin=2,
        limit=getattr(request, "param", t.BubbleLimits(10, 11)),
        actor=t.SocialAgentActor(
            name="zoo-car", agent_locator="zoo.policies:keep-lane-agent-v0"
        ),
    )


@pytest.fixture
def scenarios(bubble):
    with temp_scenario(name="straight", map="maps/straight.net.xml") as scenario_root:
        gen_scenario(
            t.Scenario(traffic={}, bubbles=[bubble]),
            output_dir=scenario_root,
        )
        yield Scenario.variations_for_all_scenario_roots([str(scenario_root)], [])


@pytest.fixture
def mock_provider():
    return MockProvider()


@pytest.fixture
def smarts(scenarios, mock_provider, time_resolution):
    smarts_ = SMARTS(
        agent_interfaces={},
        traffic_sim=SumoTrafficSimulation(
            time_resolution=time_resolution,
        ),
    )
    smarts_.add_provider(mock_provider)
    smarts_.reset(next(scenarios))
    yield smarts_
    smarts_.destroy()


def test_bubble_manager_state_change(smarts, mock_provider):
    index = smarts.vehicle_index

    vehicle_id = "vehicle"
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
        mock_provider.override_next_provider_state(
            vehicles=[
                (vehicle_id, Pose.from_center(position, Heading(-math.pi / 2)), 10)
            ]
        )

        # Providers must be disjoint
        if index.vehicle_is_hijacked(vehicle_id):
            mock_provider.clear_next_provider_state()

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


@pytest.mark.parametrize("bubble", [t.BubbleLimits(1, 1)], indirect=True)
def test_bubble_manager_limit(smarts, mock_provider, time_resolution):
    vehicle_ids = ["vehicle-1", "vehicle-2", "vehicle-3"]
    speed = 2.5
    distance_per_step = speed * time_resolution
    for x in range(200):
        vehicle_ids = {
            v_id
            for v_id in vehicle_ids
            if not smarts.vehicle_index.vehicle_is_hijacked(v_id)
        }

        vehicles = [
            (
                v_id,
                Pose.from_center(
                    (80 + y * 0.5 + x * distance_per_step, y * 4 - 4, 0),
                    Heading(-math.pi / 2),
                ),
                speed,  # speed
            )
            for y, v_id in enumerate(vehicle_ids)
        ]
        mock_provider.override_next_provider_state(vehicles=vehicles)
        smarts.step({})

    # 3 total vehicles, 1 hijacked and removed according to limit, 2 remaining
    assert (
        len(vehicle_ids) == 2
    ), "Only 1 vehicle should have been hijacked according to the limit"


def test_vehicle_spawned_in_bubble_is_not_captured(smarts, mock_provider):
    # Spawned inside bubble, didn't "drive through" airlocking region, so should _not_
    # get captured
    vehicle_id = "vehicle"
    for x in range(20):
        mock_provider.override_next_provider_state(
            vehicles=[
                (
                    vehicle_id,
                    Pose.from_center((100 + x, 0, 0), Heading(-math.pi / 2)),
                    10,  # speed
                )
            ]
        )
        smarts.step({})
        assert not smarts.vehicle_index.vehicle_is_hijacked(vehicle_id)


def test_vehicle_spawned_outside_bubble_is_captured(smarts, mock_provider):
    # Spawned vehicle drove through airlock so _should_ get captured
    vehicle_id = "vehicle"
    got_hijacked = False
    for x in range(20):
        mock_provider.override_next_provider_state(
            vehicles=[
                (
                    vehicle_id,
                    Pose.from_center((90 + x, 0, 0), Heading(-math.pi / 2)),
                    10,  # speed
                )
            ]
        )
        smarts.step({})
        if smarts.vehicle_index.vehicle_is_hijacked(vehicle_id):
            got_hijacked = True
            break

    assert got_hijacked
