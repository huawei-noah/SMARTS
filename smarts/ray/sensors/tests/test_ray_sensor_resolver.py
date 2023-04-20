# MIT License
#
# Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.
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
import pytest

from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.controllers.action_space_type import ActionSpaceType
from smarts.core.plan import Mission
from smarts.core.scenario import Scenario
from smarts.core.sensors.local_sensor_resolver import LocalSensorResolver
from smarts.core.simulation_frame import SimulationFrame
from smarts.core.simulation_local_constants import SimulationLocalConstants
from smarts.core.smarts import SMARTS
from smarts.core.sumo_traffic_simulation import SumoTrafficSimulation
from smarts.core.utils.logging import diff_unpackable
from smarts.ray.sensors.ray_sensor_resolver import RaySensorResolver

AGENT_IDS = [f"agent-00{i}" for i in range(3)]


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
        traffic_sims=[SumoTrafficSimulation()],
        envision=None,
    )
    smarts.reset(scenario)
    smarts.step({aid: [0, 0, 0] for aid in AGENT_IDS})
    yield smarts
    smarts.destroy()


def test_sensor_parallelization(
    sim: SMARTS,
):
    del sim.cached_frame
    simulation_frame: SimulationFrame = sim.cached_frame
    simulation_local_constants: SimulationLocalConstants = sim.local_constants

    parallel_resolver = RaySensorResolver(process_count_override=1)
    serial_resolver = LocalSensorResolver()

    parallel_resolver.get_ray_worker_actors(1)

    p_observations, p_dones, p_updated_sensors = parallel_resolver.observe(
        sim_frame=simulation_frame,
        sim_local_constants=simulation_local_constants,
        agent_ids=simulation_frame.agent_ids,
        renderer=sim.renderer,
        bullet_client=sim.bc,
    )
    assert len(simulation_frame.agent_ids)

    l_observations, l_dones, l_updated_sensors = serial_resolver.observe(
        sim_frame=simulation_frame,
        sim_local_constants=simulation_local_constants,
        agent_ids=simulation_frame.agent_ids,
        renderer=sim.renderer,
        bullet_client=sim.bc,
        debug=True,
    )
    assert len(simulation_frame.agent_ids)

    assert diff_unpackable(p_observations, l_observations) == ""
    assert diff_unpackable(p_dones, l_dones) == ""
    assert p_updated_sensors.keys() == l_updated_sensors.keys()
    assert set(p_updated_sensors.keys()) not in simulation_frame.agent_ids

    # TODO: Make sure that all mutable sensors are returned
    for agent_id, p_sensors in p_updated_sensors.items():
        assert p_sensors.keys() == l_updated_sensors[agent_id].keys()
        for k in p_sensors:
            assert p_sensors[k] == l_updated_sensors[agent_id][k]
