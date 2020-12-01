from smarts.core.agent_interface import AgentInterface, AgentType
import pytest
from pathlib import Path
import smarts.sstudio.types as t
from smarts.core.coordinates import Heading, Pose
from smarts.core.scenario import Scenario
from smarts.core.smarts import SMARTS
from smarts.core.sumo_traffic_simulation import SumoTrafficSimulation
from smarts.core.tests.helpers.providers import MockProvider
from smarts.sstudio import gen_scenario

from helpers.scenario import temp_scenario


# TODO: Add test for small road


AGENT_ID = "006"


@pytest.fixture
def scenarios():
    with temp_scenario(name="straight", map="maps/straight.net.xml") as scenario_root:
        traffic = t.Traffic(
            flows=[
                t.Flow(
                    route=t.Route(begin=("west", 1, 0), end=("east", 1, "max"),),
                    rate=50,
                    actors={t.TrafficActor("car"): 1},
                )
            ]
        )

        scenario = t.Scenario(
            traffic={"all": traffic},
            ego_missions=[
                t.Mission(
                    t.Route(begin=("west", 1, 10), end=("east", 1, "max")),
                    entry_tactic=t.TrapEntryTactic(
                        wait_to_hijack_limit_s=3,
                        zone=t.MapZone(start=("west", 0, 1), length=100, n_lanes=3),
                    ),
                )
            ],
        )

        gen_scenario(scenario, output_dir=scenario_root)
        yield Scenario.variations_for_all_scenario_roots(
            [str(scenario_root)], [AGENT_ID]
        )


@pytest.fixture
def empty_scenarios():
    with temp_scenario(name="straight", map="maps/straight.net.xml") as scenario_root:
        scenario = t.Scenario(
            traffic={},
            ego_missions=[
                t.Mission(
                    t.Route(begin=("west", 1, 10), end=("east", 1, "max")),
                    entry_tactic=t.TrapEntryTactic(
                        wait_to_hijack_limit_s=3,
                        zone=t.MapZone(start=("west", 0, 1), length=100, n_lanes=3),
                    ),
                )
            ],
        )

        gen_scenario(scenario, output_dir=scenario_root)
        yield Scenario.variations_for_all_scenario_roots(
            [str(scenario_root)], [AGENT_ID]
        )


@pytest.fixture
def no_agent_capture_scenarios():
    with temp_scenario(name="straight", map="maps/straight.net.xml") as scenario_root:
        traffic = t.Traffic(
            flows=[
                t.Flow(
                    route=t.Route(begin=("west", 1, 0), end=("east", 1, "max"),),
                    rate=50,
                    actors={t.TrafficActor("car"): 1},
                )
            ]
        )

        scenario = t.Scenario(
            traffic={"all": traffic},
            ego_missions=[
                t.Mission(
                    t.Route(begin=("west", 1, 10), end=("east", 1, "max")),
                    entry_tactic=t.TrapEntryTactic(
                        wait_to_hijack_limit_s=3,
                        zone=t.MapZone(start=("west", 0, 1), length=100, n_lanes=3),
                    ),
                ),
                t.Mission(
                    t.Route(begin=("west", 1, 10), end=("east", 1, "max")),
                    entry_tactic=t.TrapEntryTactic(
                        wait_to_hijack_limit_s=3.1,
                        zone=t.MapZone(start=("west", 0, 1), length=100, n_lanes=3),
                    ),
                ),
            ],
        )

        gen_scenario(scenario, output_dir=scenario_root)
        yield Scenario.variations_for_all_scenario_roots(
            [str(scenario_root)], [AGENT_ID, AGENT_ID + "1"]
        )


@pytest.fixture
def smarts():
    smarts_ = SMARTS(
        agent_interfaces={
            AGENT_ID: AgentInterface.from_type(AgentType.Laner, max_episode_steps=30)
        },
        traffic_sim=SumoTrafficSimulation(time_resolution=0.1),
    )
    yield smarts_
    smarts_.destroy()


def test_capture_vehicle(smarts: SMARTS, scenarios):
    vehicle_prefix = "car-flow-route-west_1_0-east_1_max"
    smarts.reset(next(scenarios))

    print(smarts.vehicle_index.agent_vehicle_ids)
    vehicle_id = list(smarts.vehicle_index.agent_vehicle_ids)[0]
    assert vehicle_id.startswith(vehicle_prefix)
    assert smarts.elapsed_sim_time < 1
    assert len(smarts.vehicle_index.agent_vehicle_ids) == 1
    assert smarts.vehicle_index.actor_id_from_vehicle_id(vehicle_id).startswith(
        AGENT_ID
    )


def test_emit_on_default(smarts: SMARTS, empty_scenarios):
    smarts.reset(next(empty_scenarios))
    assert smarts.elapsed_sim_time >= 3
    assert len(smarts.vehicle_index.agent_vehicle_ids) == 1
    assert len(smarts.vehicle_index.vehicle_ids_by_actor_id(AGENT_ID)) == 1


# def test_no_recapture_agent(smarts: SMARTS, no_agent_capture_scenarios):
#     smarts.reset(next(no_agent_capture_scenarios))
#     smarts.step({})
#     smarts.step({})
#     assert len(smarts.vehicle_index.agent_vehicle_ids) > 1
