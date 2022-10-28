# MIT License
#
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
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
import logging
from typing import Any, Dict, List, Set

import pytest
from helpers.scenario import maps_dir

from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.controllers import ActionSpaceType
from smarts.core.plan import Mission
from smarts.core.road_map import RoadMap
from smarts.core.scenario import Scenario
from smarts.core.sensors import Observation, Sensors, SensorState, SensorsWorker
from smarts.core.smarts import SMARTS, SimulationFrame
from smarts.core.sumo_traffic_simulation import SumoTrafficSimulation
from smarts.core.utils.file import unpack

SimulationState = SimulationFrame
SensorState = Any

AGENT_IDS = [f"agent-00{i}" for i in range(30)]


@pytest.fixture
def agents_to_be_briefed():
    return AGENT_IDS


@pytest.fixture
def scenario(agents_to_be_briefed: List[str]) -> Scenario:
    s = Scenario(
        scenario_root="scenarios/sumo/loop",
        traffic_specs=["scenarios/sumo/loop/traffic/basic.rou.xml"],
        missions=dict(
            zip(
                agents_to_be_briefed,
                Scenario.discover_agent_missions(
                    scenario_root="scenarios/sumo/loop",
                    agents_to_be_briefed=agents_to_be_briefed,
                ),
            )
        ),
    )
    missions = [
        Mission.random_endless_mission(
            s.road_map,
        )
        for a in agents_to_be_briefed
    ]
    s.set_ego_missions(dict(zip(agents_to_be_briefed, missions)))
    return s


@pytest.fixture()
def sim(scenario):
    # agents = {aid: AgentInterface.from_type(AgentType.Full) for aid in AGENT_IDS},
    agents = {
        aid: AgentInterface.from_type(
            AgentType.Buddha, action=ActionSpaceType.Continuous
        )
        for aid in AGENT_IDS
    }
    smarts = SMARTS(
        agents,
        traffic_sims=[SumoTrafficSimulation(headless=True)],
        envision=None,
    )
    smarts.reset(scenario)
    smarts.step({aid: [0, 0, 0] for aid in AGENT_IDS})
    yield smarts
    smarts.destroy()


@pytest.fixture()
def simulation_frame(sim) -> SimulationState:
    frame = sim.frame()
    yield frame


@pytest.fixture
def vehicle_ids():
    yield {}


@pytest.fixture
def renderer_type():
    yield None


def test_sensor_parallelization(
    simulation_frame: SimulationState,
):

    import time

    # Sensors.init(road_map, renderer_type)  # not required
    agent_ids = set(AGENT_IDS)
    non_parallel_start = time.monotonic()
    obs, dones = Sensors.observe_parallel(
        simulation_frame, agent_ids, process_count_override=0
    )
    non_parallel_total = time.monotonic() - non_parallel_start

    parallel_1_start = time.monotonic()
    obs, dones = Sensors.observe_parallel(
        simulation_frame, agent_ids, process_count_override=1
    )
    parallel_1_total = time.monotonic() - parallel_1_start

    parallel_2_start = time.monotonic()
    obs, dones = Sensors.observe_parallel(
        simulation_frame, agent_ids, process_count_override=2
    )
    parallel_2_total = time.monotonic() - parallel_2_start

    parallel_4_start = time.monotonic()
    obs, dones = Sensors.observe_parallel(
        simulation_frame, agent_ids, process_count_override=4
    )
    parallel_4_total = time.monotonic() - parallel_4_start

    assert len(obs) > 0
    assert (
        non_parallel_total > parallel_1_total
        and parallel_1_total > parallel_2_total
        and parallel_2_total > parallel_4_total
    ), f"{non_parallel_total=}, {parallel_1_total=}, {parallel_2_total=}, {parallel_4_total=}"


def test_sensor_worker(
    simulation_frame: SimulationState,
):
    agent_ids = set(AGENT_IDS)
    worker = SensorsWorker()
    worker.run()
    worker.send_to_process(simulation_frame, agent_ids)
    observations, dones = SensorsWorker.local(simulation_frame, agent_ids)
    other_observations, other_dones = worker.result(block=True)

    assert isinstance(observations, dict)
    assert all(
        [isinstance(obs, Observation) for obs in observations.values()]
    ), f"{observations=}"
    assert isinstance(dones, dict)
    assert all([isinstance(obs, bool) for obs in dones.values()])
    assert isinstance(other_observations, dict)
    assert all([isinstance(obs, Observation) for obs in other_observations.values()])
    assert isinstance(other_dones, dict)
    assert all([isinstance(obs, bool) for obs in other_dones.values()])
    assert str(unpack(observations)) == str(unpack(other_observations))
