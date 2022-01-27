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
import pytest
from helpers.scenario import temp_scenario

import smarts.sstudio.types as t
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.scenario import Scenario
from smarts.core.smarts import SMARTS
from smarts.core.sumo_traffic_simulation import SumoTrafficSimulation
from smarts.sstudio import gen_scenario

AGENT_ID = "006"
AGENT_ID_2 = "007"


@pytest.fixture
def scenarios():
    with temp_scenario(name="straight", map="maps/straight.net.xml") as scenario_root:
        traffic = t.Traffic(
            flows=[
                t.Flow(
                    route=t.Route(
                        begin=("west", 1, 0),
                        end=("east", 1, "max"),
                    ),
                    rate=50,
                    actors={t.TrafficActor("car"): 1},
                )
            ]
        )

        missions = [
            t.Mission(
                t.Route(begin=("west", 1, 10), end=("east", 1, "max")),
                entry_tactic=t.TrapEntryTactic(
                    wait_to_hijack_limit_s=3,
                    zone=t.MapZone(start=("west", 0, 1), length=100, n_lanes=3),
                ),
            )
        ]
        scenario = t.Scenario(
            traffic={"all": traffic},
            ego_missions=missions,
        )

        gen_scenario(scenario, output_dir=scenario_root)
        yield Scenario.variations_for_all_scenario_roots(
            [str(scenario_root)], [AGENT_ID]
        )


@pytest.fixture
def empty_scenarios():
    with temp_scenario(name="straight", map="maps/straight.net.xml") as scenario_root:
        missions = [
            t.Mission(
                t.Route(begin=("west", 1, 10), end=("east", 1, "max")),
                entry_tactic=t.TrapEntryTactic(
                    wait_to_hijack_limit_s=3,
                    zone=t.MapZone(start=("west", 0, 1), length=100, n_lanes=3),
                ),
            )
        ]
        scenario = t.Scenario(
            traffic={},
            ego_missions=missions,
        )

        gen_scenario(scenario, output_dir=scenario_root)
        yield Scenario.variations_for_all_scenario_roots(
            [str(scenario_root)], [AGENT_ID]
        )


@pytest.fixture
def two_agent_capture_offset_tenth_of_second():
    with temp_scenario(name="straight", map="maps/straight.net.xml") as scenario_root:
        missions = [
            t.Mission(
                t.Route(begin=("west", 1, 20), end=("east", 1, "max")),
                entry_tactic=t.TrapEntryTactic(
                    wait_to_hijack_limit_s=0,
                    zone=t.MapZone(start=("west", 0, 1), length=100, n_lanes=3),
                ),
            ),
            t.Mission(
                t.Route(begin=("west", 2, 10), end=("east", 1, "max")),
                entry_tactic=t.TrapEntryTactic(
                    wait_to_hijack_limit_s=0.1,
                    zone=t.MapZone(start=("west", 0, 1), length=100, n_lanes=3),
                ),
            ),
        ]

        scenario = t.Scenario(
            traffic={},
            ego_missions=missions,
        )

        gen_scenario(scenario, output_dir=scenario_root)
        yield Scenario.variations_for_all_scenario_roots(
            [str(scenario_root)], [AGENT_ID, AGENT_ID_2]
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


@pytest.fixture
def smarts_two_agents():
    smarts_ = SMARTS(
        agent_interfaces={
            AGENT_ID: AgentInterface.from_type(AgentType.Laner, max_episode_steps=30),
            AGENT_ID_2: AgentInterface.from_type(AgentType.Laner, max_episode_steps=30),
        },
        traffic_sim=SumoTrafficSimulation(time_resolution=0.1),
    )
    yield smarts_
    smarts_.destroy()


def test_capture_vehicle(smarts: SMARTS, scenarios):
    vehicle_prefix = "car-flow-route-west_1_0-east_1_max"
    smarts.reset(next(scenarios))

    vehicle_id = list(smarts.vehicle_index.agent_vehicle_ids())[0]
    assert vehicle_id.startswith(vehicle_prefix)
    assert smarts.elapsed_sim_time < 1
    assert len(smarts.vehicle_index.agent_vehicle_ids()) == 1
    assert smarts.vehicle_index.actor_id_from_vehicle_id(vehicle_id).startswith(
        AGENT_ID
    )


def test_emit_on_default(smarts: SMARTS, empty_scenarios):
    smarts.reset(next(empty_scenarios))
    assert round(smarts.elapsed_sim_time, 2) == 3.1
    assert len(smarts.vehicle_index.agent_vehicle_ids()) == 1
    assert len(smarts.vehicle_index.vehicle_ids_by_actor_id(AGENT_ID)) == 1


def test_no_recapture_agent(
    smarts_two_agents: SMARTS, two_agent_capture_offset_tenth_of_second
):
    smarts_two_agents.reset(next(two_agent_capture_offset_tenth_of_second))
    for i in range(3):
        smarts_two_agents.step({})
    assert len(smarts_two_agents.agent_manager.pending_agent_ids) == 0
    assert len(smarts_two_agents.agent_manager.active_agents) == 2
    assert len(smarts_two_agents.vehicle_index.agent_vehicle_ids()) == 2
