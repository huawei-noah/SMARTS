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
from unittest import mock

import pytest

from smarts.core.actor import ActorState
from smarts.core.agent_interface import (
    AgentInterface,
    AgentsAliveDoneCriteria,
    AgentsListAlive,
    DoneCriteria,
    InterestDoneCriteria,
)
from smarts.core.scenario import Scenario
from smarts.core.sensors import Sensors
from smarts.core.smarts import SMARTS, SimulationFrame
from smarts.core.sumo_traffic_simulation import SumoTrafficSimulation

AGENT1 = "agent1"
AGENT2 = "agent2"
AGENT3 = "agent3"


@pytest.fixture
def scenario():
    scenario = Scenario(
        scenario_root="scenarios/sumo/loop",
        traffic_specs=["scenarios/sumo/loop/build/traffic/basic.rou.xml"],
    )
    return scenario


ego_alive_test = AgentsAliveDoneCriteria(
    minimum_ego_agents_alive=2,
)

custom_agent_list_test = AgentsAliveDoneCriteria(
    agent_lists_alive=[
        AgentsListAlive(
            agents_list=[AGENT1, AGENT2], minimum_agents_alive_in_list=1
        ),  # requires 1 of AGENT1 and AGENT2 to be alive
        AgentsListAlive(
            agents_list=[AGENT3], minimum_agents_alive_in_list=1
        ),  # requires AGENT3 to be alive
    ]
)

total_alive_test = AgentsAliveDoneCriteria(
    minimum_total_agents_alive=2,
)


@pytest.fixture(
    scope="module", params=[ego_alive_test, custom_agent_list_test, total_alive_test]
)
def sim(request):
    shared_interface = AgentInterface(
        done_criteria=DoneCriteria(agents_alive=request.param)
    )
    agents = {
        AGENT1: shared_interface,
        AGENT2: shared_interface,
        AGENT3: shared_interface,
    }
    smarts = SMARTS(
        agents,
        traffic_sims=[SumoTrafficSimulation(headless=True)],
        envision=None,
    )

    yield smarts
    smarts.destroy()


def test_agents_alive_done_check(sim, scenario):
    sim.setup(scenario)
    interface = sim.agent_manager.agent_interface_for_agent_id(AGENT1)
    done_criteria = interface.done_criteria

    sim_frame: SimulationFrame = sim.cached_frame
    # 3 agents available, done requires 2 to be alive
    assert not Sensors._agents_alive_done_check(
        sim_frame.ego_ids, sim_frame.potential_agent_ids, done_criteria.agents_alive
    )

    sim.agent_manager.teardown_ego_agents({AGENT2})
    del sim.cached_frame
    sim_frame: SimulationFrame = sim.cached_frame
    # 2 agents available, done requires 2 to be alive
    assert not Sensors._agents_alive_done_check(
        sim_frame.ego_ids, sim_frame.potential_agent_ids, done_criteria.agents_alive
    )

    sim.agent_manager.teardown_ego_agents({AGENT3})
    del sim.cached_frame
    sim_frame: SimulationFrame = sim.cached_frame
    # 1 agents available, done requires 2 to be alive
    assert Sensors._agents_alive_done_check(
        sim_frame.ego_ids, sim_frame.potential_agent_ids, done_criteria.agents_alive
    )

    sim.agent_manager.teardown_ego_agents({AGENT1})
    del sim.cached_frame
    sim_frame: SimulationFrame = sim.cached_frame
    # 1 agents available, done requires 2 to be alive
    assert Sensors._agents_alive_done_check(
        sim_frame.ego_ids, sim_frame.potential_agent_ids, done_criteria.agents_alive
    )


def test_interest_done():
    waiting_interest_criteria = InterestDoneCriteria(("leader",), strict=False)
    strict_interest_criteria = InterestDoneCriteria(("leader",), strict=True)

    sensor_state = mock.Mock()
    sensor_state.seen_interest_actors = True
    # 0 interest vehicles, previously seen, done requires 1 to exist
    assert Sensors._interest_done_check({}, sensor_state, strict_interest_criteria)
    assert Sensors._interest_done_check({}, sensor_state, waiting_interest_criteria)

    sensor_state = mock.Mock()
    sensor_state.seen_interest_actors = False
    # 0 interest vehicles, previously seen, done requires 1 to exist
    assert Sensors._interest_done_check({}, sensor_state, strict_interest_criteria)
    assert not Sensors._interest_done_check({}, sensor_state, waiting_interest_criteria)

    sensor_state = mock.Mock()
    sensor_state.seen_interest_actors = False
    # 0 interest vehicles, done requires 1 to exist
    assert not Sensors._interest_done_check(
        {"leader": ActorState("leader")}, sensor_state, strict_interest_criteria
    )
    assert not Sensors._interest_done_check(
        {"leader": ActorState("leader")}, sensor_state, waiting_interest_criteria
    )
