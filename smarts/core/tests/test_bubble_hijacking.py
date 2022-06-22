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
from collections import defaultdict
from dataclasses import dataclass

import pytest
import re
from helpers.bubbles import bubble_geometry

# TODO: Rename temp_scenario(...)
from helpers.scenario import temp_scenario
from shapely.geometry import Point

import smarts.sstudio.types as t
from smarts.core.local_traffic_provider import LocalTrafficProvider
from smarts.core.scenario import Scenario
from smarts.core.smarts import SMARTS
from smarts.core.sumo_traffic_simulation import SumoTrafficSimulation
from smarts.sstudio import gen_scenario


@pytest.fixture
def num_vehicles():
    return 3


@pytest.fixture
def bubbles():
    actor = t.SocialAgentActor(
        name="keep-lane-agent",
        agent_locator="zoo.policies:keep-lane-agent-v0",
    )

    return [
        t.Bubble(
            zone=t.PositionalZone(pos=(150, -60), size=(30, 30)),
            margin=10,
            actor=actor,
        ),
        t.Bubble(
            zone=t.PositionalZone(pos=(60, -60), size=(30, 30)),
            margin=10,
            actor=actor,
        ),
    ]


@pytest.fixture(params=["SUMO", "SMARTS"])
def traffic_sim(request):
    return getattr(request, "param", "SUMO")


@pytest.fixture
def scenarios(bubbles, num_vehicles, traffic_sim):
    with temp_scenario(name="6lane", map="maps/6lane.net.xml") as scenario_root:
        traffic = t.Traffic(
            engine=traffic_sim,
            flows=[
                t.Flow(
                    route=t.Route(
                        begin=("edge-west-WE", lane, 0),
                        end=("edge-east-WE", lane, "max"),
                    ),
                    rate=10,
                    actors={t.TrafficActor(name="car"): 1},
                )
                for lane in range(num_vehicles)
            ],
        )

        gen_scenario(
            t.Scenario(traffic={"all": traffic}, bubbles=bubbles),
            output_dir=scenario_root,
        )

        yield Scenario.variations_for_all_scenario_roots([str(scenario_root)], [])


@pytest.fixture
def smarts(traffic_sim):
    traffic_sims = (
        [LocalTrafficProvider()]
        if traffic_sim == "SMARTS"
        else [SumoTrafficSimulation()]
    )
    smarts = SMARTS({}, traffic_sims=traffic_sims)
    yield smarts
    smarts.destroy()


@dataclass
class ZoneSteps:
    in_bubble: int = 0
    outside_bubble: int = 0
    airlock_entry: int = 0
    airlock_exit: int = 0


# TODO: Consider a higher-level DSL syntax to fulfill these tests
@pytest.mark.parametrize("traffic_sim", ["SUMO", "SMARTS"], indirect=True)
def test_bubble_hijacking(smarts, scenarios, bubbles, num_vehicles, traffic_sim):
    """Ensures bubble airlocking, hijacking, and relinquishing are functional.
    Additionally, we test with multiple bubbles and vehicles to ensure operation is
    correct in these conditions as well.
    """
    scenario = next(scenarios)
    smarts.reset(scenario)

    index = smarts.vehicle_index
    geometries = [bubble_geometry(b, smarts.road_map) for b in bubbles]

    # bubble: vehicle: steps per zone
    steps_driven_in_zones = {b.id: defaultdict(lambda: ZoneSteps()) for b in bubbles}
    vehicles_made_to_through_bubble = {b.id: [] for b in bubbles}
    for _ in range(300):
        smarts.step({})
        for vehicle in index.vehicles:
            position = Point(vehicle.position)
            for bubble, geometry in zip(bubbles, geometries):
                in_bubble = position.within(geometry.bubble)
                is_shadowing = (
                    index.shadow_actor_id_from_vehicle_id(vehicle.id) is not None
                )
                is_agent_controlled = vehicle.id in index.agent_vehicle_ids()

                vehicle_id = (
                    vehicle.id
                    if traffic_sim == "SUMO"
                    else re.sub("_\d+$", "", vehicle.id)
                )
                zone_steps = steps_driven_in_zones[bubble.id][vehicle_id]
                if position.within(geometry.bubble):
                    zone_steps.in_bubble += 1
                    assert in_bubble and not is_shadowing and is_agent_controlled
                elif position.within(geometry.airlock_entry):
                    zone_steps.airlock_entry += 1
                    assert not in_bubble and is_shadowing and not is_agent_controlled
                elif position.within(geometry.airlock_exit):
                    zone_steps.airlock_exit += 1
                    # TODO: Presently not implemented, but `is_shadowing` should be True
                    assert not in_bubble and not is_shadowing and is_agent_controlled
                    if vehicle_id not in vehicles_made_to_through_bubble[bubble.id]:
                        vehicles_made_to_through_bubble[bubble.id].append(vehicle_id)
                elif not any([position.within(geom.airlock) for geom in geometries]):
                    # Not in any bubble; airlock is the encompassing region
                    zone_steps.outside_bubble += 1
                    assert (
                        not in_bubble and not is_shadowing and not is_agent_controlled
                    )

    # Just to have some padding, we want to be in each region at least 5 steps
    min_steps = 5
    for bubble_id, zones in steps_driven_in_zones.items():
        vehicle_ids = vehicles_made_to_through_bubble[bubble_id]
        assert (
            len(vehicle_ids) >= num_vehicles
        ), "Insufficient no. vehicles drove through bubble"
        for vehicle_id in vehicle_ids[:num_vehicles]:
            zone = zones[vehicle_id]
            assert all(
                [
                    zone.in_bubble > min_steps,
                    zone.outside_bubble > min_steps,
                    zone.airlock_entry > min_steps,
                    zone.airlock_exit > min_steps,
                ]
            ), (
                f"bubble={bubble_id}, vehicle_id={vehicle_id}, zone={zone} doesn't meet "
                f"min_steps={min_steps} requirement"
            )
