from collections import defaultdict
from dataclasses import dataclass

import pytest
from shapely.geometry import Point

import smarts.sstudio.types as t
from helpers.scenario import temp_scenario
from helpers.bubbles import bubble_geometry
from smarts.core.scenario import Scenario
from smarts.core.smarts import SMARTS
from smarts.core.sumo_traffic_simulation import SumoTrafficSimulation
from smarts.sstudio import gen_scenario


MAX_BOID_LIMIT = 3


@pytest.fixture
def bubble():
    return t.Bubble(
        zone=t.PositionalZone(pos=(100, 0), size=(20, 20)),
        margin=10,
        limit=MAX_BOID_LIMIT,
        actor=t.BoidAgentActor(
            # TODO: Provide a more self-contained way to build agent locators for tests
            name="hive-mind",
            agent_locator="scenarios.straight.agent_prefabs:pose-boid-agent-v0",
        ),
    )


@pytest.fixture
def scenarios(bubble):
    with temp_scenario(name="straight", map="maps/straight.net.xml") as scenario_root:
        traffic = t.Traffic(
            flows=[
                t.Flow(
                    route=t.Route(
                        begin=("west", lane_idx, 0), end=("east", lane_idx, "max"),
                    ),
                    rate=50,
                    actors={t.TrafficActor("car"): 1,},
                )
                for lane_idx in range(3)
            ]
        )

        gen_scenario(
            t.Scenario(traffic={"all": traffic}, bubbles=[bubble]),
            output_dir=scenario_root,
        )

        yield Scenario.variations_for_all_scenario_roots([str(scenario_root)], [])


@pytest.fixture
def smarts():
    smarts = SMARTS({}, traffic_sim=SumoTrafficSimulation())
    yield smarts
    smarts.destroy()


@dataclass
class ZoneSteps:
    in_bubble: int = 0
    outside_bubble: int = 0
    airlock_entry: int = 0
    airlock_exit: int = 0


# TODO: Consider a higher-level DSL syntax to fulfill these tests
def test_boids(smarts, scenarios, bubble):
    scenario = next(scenarios)
    smarts.reset(scenario)

    index = smarts.vehicle_index
    geometry = bubble_geometry(bubble, smarts.road_network)

    triggered_multiple_vehicles_in_bubble = False
    triggered_multiple_vehicles_airlocked = False

    # vehicle: steps per zone
    steps_driven_in_zones = defaultdict(lambda: ZoneSteps())
    # TODO: It's possible that multiple vehicles get spawned within the 500 steps but
    #       not all of them make it through the bubble completely causing the test to
    #       fail.
    for _ in range(500):
        smarts.step({})

        hijacked_actor_ids = []
        shadowed_actor_ids = []

        for vehicle in index.vehicles:
            position = Point(vehicle.position)
            in_bubble = position.within(geometry.bubble)
            is_shadowing = index.shadow_actor_id_from_vehicle_id(vehicle.id) is not None
            is_agent_controlled = vehicle.id in index.agent_vehicle_ids

            zone_steps = steps_driven_in_zones[vehicle.id]
            if position.within(geometry.bubble):
                zone_steps.in_bubble += 1
                hijacked_actor_ids.append(index.actor_id_from_vehicle_id(vehicle.id))
                assert in_bubble and not is_shadowing and is_agent_controlled
            elif position.within(geometry.airlock_entry):
                zone_steps.airlock_entry += 1
                shadowed_actor_ids.append(
                    index.shadow_actor_id_from_vehicle_id(vehicle.id)
                )
                assert not in_bubble and is_shadowing and not is_agent_controlled
            elif position.within(geometry.airlock_exit):
                zone_steps.airlock_exit += 1
                # TODO: Presently not implemented, but `is_shadowing` should be True
                assert not in_bubble and not is_shadowing and is_agent_controlled
            else:
                zone_steps.outside_bubble += 1
                assert not in_bubble and not is_shadowing and not is_agent_controlled

        if len(hijacked_actor_ids) > 1:
            triggered_multiple_vehicles_in_bubble = True

        if len(shadowed_actor_ids) > 1:
            triggered_multiple_vehicles_airlocked = True

        assert (
            len(set(hijacked_actor_ids)) <= 1
        ), "Boid vehicles must be controlled by the same actor"
        assert (
            len(set(shadowed_actor_ids)) <= 1
        ), "Boid vehicles must be shadowed by the same actor"

    # Just to have some padding, we want to be in each region at least 5 steps
    min_steps = 5
    for vehicle_id, zone in steps_driven_in_zones.items():
        assert all(
            [
                zone.in_bubble > min_steps,
                zone.outside_bubble > min_steps,
                zone.airlock_entry > min_steps,
                zone.airlock_exit > min_steps,
            ]
        ), (
            f"vehicle_id={vehicle_id}, zone={zone} doesn't meet "
            f"min_steps={min_steps} requirement"
        )

    assert (
        triggered_multiple_vehicles_in_bubble
    ), "Multiple vehicles did not enter the bubble simultaneously"
    assert (
        triggered_multiple_vehicles_airlocked
    ), "Multiple vehicles were not airlocked simultaneously"
