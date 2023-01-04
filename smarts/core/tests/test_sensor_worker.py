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
from typing import Any

import pytest

from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.controllers import ActionSpaceType
from smarts.core.plan import Mission
from smarts.core.scenario import Scenario
from smarts.core.sensors import (
    Observation,
    SensorsWorker,
    SensorsWorkerRequestId,
    WorkerKwargs,
)
from smarts.core.simulation_frame import SimulationFrame
from smarts.core.smarts import SMARTS
from smarts.core.sumo_traffic_simulation import SumoTrafficSimulation
from smarts.core.utils.logging import diff_unpackable

SimulationState = SimulationFrame
SensorState = Any

AGENT_IDS = [f"agent-00{i}" for i in range(100)]


@pytest.fixture
def scenario() -> Scenario:
    s = Scenario(
        scenario_root="scenarios/sumo/loop",
        traffic_specs=["scenarios/sumo/loop/build/traffic/basic.rou.xml"],
        missions=dict(
            zip(
                AGENT_IDS,
                Scenario.discover_agent_missions(
                    scenario_root="scenarios/sumo/loop",
                    agents_to_be_briefed=AGENT_IDS,
                ),
            )
        ),
    )
    missions = [
        Mission.random_endless_mission(
            s.road_map,
        )
        for _ in AGENT_IDS
    ]
    s.set_ego_missions(dict(zip(AGENT_IDS, missions)))
    return s


@pytest.fixture(
    params=[
        AgentInterface.from_type(
            AgentType.Laner,
            action=ActionSpaceType.Continuous,
        ),
        AgentInterface.from_type(
            AgentType.Full,
            action=ActionSpaceType.Continuous,
        ),
    ]
)
def sim(scenario, request):
    a_interface = getattr(request, "param")
    agents = {aid: a_interface for aid in AGENT_IDS}
    smarts = SMARTS(
        agents,
        traffic_sims=[SumoTrafficSimulation(headless=True)],
        envision=None,
    )
    smarts.reset(scenario)
    smarts.step({aid: [0, 0, 0] for aid in AGENT_IDS})
    yield smarts
    smarts.destroy()


def test_sensor_worker(
    sim: SMARTS,
):
    del sim.cached_frame
    sim_frame: SimulationFrame = sim.cached_frame
    agent_ids = set(AGENT_IDS)
    worker = SensorsWorker()
    worker.run()
    worker.send(
        request=SensorsWorker.Request(
            SensorsWorkerRequestId.SIMULATION_LOCAL_CONSTANTS,
            WorkerKwargs(sim_local_constants=sim.local_constants),
        )
    )
    assert worker.running
    worker_args = WorkerKwargs(sim_frame=sim_frame, agent_ids=agent_ids)
    worker.send(
        SensorsWorker.Request(SensorsWorkerRequestId.SIMULATION_FRAME, worker_args)
    )
    state = dict(
        sim_frame=sim_frame,
        sim_local_constants=sim.local_constants,
        agent_ids=agent_ids,
    )
    observations, dones, updated_sensors = SensorsWorker.local(state=state)
    other_observations, other_dones, updated_sensors = worker.result(timeout=5)

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
    assert observations.keys() == other_dones.keys()
    assert diff_unpackable(other_observations, observations) == ""
