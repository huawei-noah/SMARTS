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
from smarts.core.sensors import (
    Observation,
    Sensors,
    SensorState,
    SensorsWorker,
    WorkerKwargs,
)
from smarts.core.simulation_frame import SimulationFrame
from smarts.core.simulation_local_constants import SimulationLocalConstants
from smarts.core.smarts import SMARTS
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
def sim(scenario) -> SMARTS:
    # agents = {aid: AgentInterface.from_type(AgentType.Full) for aid in AGENT_IDS},
    agents = {
        aid: AgentInterface.from_type(
            AgentType.Full,
            drivable_area_grid_map=False,
            ogm=False,
            rgb=False,
            lidar=False,
            action=ActionSpaceType.Continuous,
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
    frame = sim.cached_frame
    yield frame


@pytest.fixture
def vehicle_ids():
    yield {}


@pytest.fixture
def renderer_type():
    yield None


def test_sensor_parallelization(
    sim: SMARTS,
):
    import time

    del sim.cached_frame
    simulation_frame: SimulationFrame = sim.cached_frame
    simulation_local_constants: SimulationLocalConstants = sim.local_constants

    def observe_with_processes(processes):
        start_time = time.monotonic()
        obs, dones = Sensors.observe_parallel(
            sim_frame=simulation_frame,
            sim_local_constants=simulation_local_constants,
            agent_ids=simulation_frame.agent_ids,
            renderer=sim.renderer,
            process_count_override=processes,
        )
        assert len(obs) > 0
        return time.monotonic() - start_time

    sensors_instance = Sensors.instance()
    sensors_instance.get_workers(4, sim_local_constants=sim.local_constants)

    time.sleep(0.5)

    serial_total = observe_with_processes(0)
    parallel_1_total = observe_with_processes(1)
    parallel_2_total = observe_with_processes(2)
    parallel_3_total = observe_with_processes(3)
    parallel_4_total = observe_with_processes(4)

    assert (
        serial_total > parallel_1_total
        or serial_total > parallel_2_total
        or serial_total > parallel_3_total
    ), f"{serial_total}, {parallel_1_total}, {parallel_2_total}, {parallel_3_total} {parallel_4_total}"


def test_sensor_worker(
    sim: SMARTS,
):
    del sim.cached_frame
    simulation_frame: SimulationFrame = sim.cached_frame
    agent_ids = set(AGENT_IDS)
    worker = SensorsWorker()
    worker.run(sim_local_constants=sim.local_constants)
    worker_args = WorkerKwargs(sim_frame=simulation_frame)
    worker.send_to_process(worker_args=worker_args, agent_ids=agent_ids)
    observations, dones = SensorsWorker.local(
        simulation_frame, sim.local_constants, agent_ids
    )
    other_observations, other_dones = worker.result(block=True, timeout=5)

    assert isinstance(observations, dict)
    assert all(
        [isinstance(obs, Observation) for obs in observations.values()]
    ), f"{observations}"
    assert isinstance(dones, dict)
    assert all([isinstance(obs, bool) for obs in dones.values()])
    assert isinstance(other_observations, dict)
    assert all([isinstance(obs, Observation) for obs in other_observations.values()])
    assert isinstance(other_dones, dict)
    assert all([isinstance(obs, bool) for obs in other_dones.values()])
    assert str(unpack(observations)) == str(unpack(other_observations))
