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
from typing import List

import pytest

import smarts.core.serialization.default as serializer
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.scenario import Scenario
from smarts.core.smarts import SMARTS, SimulationFrame
from smarts.core.sumo_traffic_simulation import SumoTrafficSimulation

AGENT_ID = "agent-007"


@pytest.fixture
def agents_to_be_briefed():
    return [AGENT_ID]


@pytest.fixture()
def sim():
    agents = {AGENT_ID: AgentInterface.from_type(AgentType.Full)}
    smarts = SMARTS(
        agents,
        traffic_sims=[SumoTrafficSimulation(headless=True)],
        envision=None,
    )

    yield smarts
    smarts.destroy()


@pytest.fixture
def scenario(agents_to_be_briefed: List[str]) -> Scenario:
    return Scenario(
        scenario_root="scenarios/sumo/loop",
        traffic_specs=["scenarios/sumo/loop/build/traffic/basic.rou.xml"],
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


def test_state(sim: SMARTS, scenario):
    sim.setup(scenario)
    frame: SimulationFrame = sim.cached_frame

    assert isinstance(frame, SimulationFrame)
    assert hasattr(frame, "actor_states")
    assert hasattr(frame, "agent_interfaces")
    assert hasattr(frame, "agent_vehicle_controls")
    assert hasattr(frame, "ego_ids")
    assert hasattr(frame, "elapsed_sim_time")
    assert hasattr(frame, "fixed_timestep")
    assert hasattr(frame, "resetting")
    # This attribute was temporarily added but should not be here
    assert not hasattr(frame, "road_map")

    assert hasattr(frame, "map_spec")
    assert hasattr(frame, "last_dt")
    assert hasattr(frame, "last_provider_state")
    assert hasattr(frame, "step_count")
    assert hasattr(frame, "vehicle_collisions")


def test_vehicles_in_actors(sim: SMARTS, scenario):
    sim.setup(scenario)
    frame: SimulationFrame = sim.cached_frame

    while (frame := sim.cached_frame) and len(frame.vehicle_states) < 1:
        sim.step({})

    assert set(k for k in frame.vehicle_states) == set(
        actor_state.actor_id for actor_state in frame.actor_states
    )
    actor_states = {
        actor_state.actor_id: actor_state for actor_state in frame.actor_states
    }
    for k, vehicle_state in frame.vehicle_states.items():
        assert vehicle_state == actor_states[k]


def test_state_serialization(sim: SMARTS, scenario: Scenario):
    sim.setup(scenario)
    sim.reset(scenario, start_time=10)
    sim.step({AGENT_ID: [0, 0, 0]})
    frame: SimulationFrame = sim.cached_frame

    # Check if serialization will work
    serialized = serializer.dumps(frame)
    deserialized: SimulationFrame = serializer.loads(serialized)

    # dataclass allows comparison
    assert frame == deserialized
