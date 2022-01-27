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

import smarts.sstudio.types as t
from smarts.core.agent import Agent, AgentSpec
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.controllers import LaneFollowingController
from smarts.core.scenario import Scenario
from smarts.core.smarts import SMARTS
from smarts.core.sumo_traffic_simulation import SumoTrafficSimulation
from smarts.core.tests.helpers.scenario import temp_scenario
from smarts.sstudio import gen_scenario

AGENT_ID = "Agent-007"

# Tests are parameterized based on lane following agent types
# and speeds. Note that for "Laner" type the speed is fixed.
@pytest.fixture(
    params=[
        ("keep_lane", AgentType.Laner),
        ((10, 0), AgentType.LanerWithSpeed),
        ((16, 0), AgentType.LanerWithSpeed),
    ]
)
def agent_and_agent_type(request):
    class FixedAgent(Agent):
        def __init__(self, action=request.param[0]):
            self.action = action

        def act(self, obs):
            return self.action

    return (FixedAgent, request.param[1])


@pytest.fixture(
    params=[
        ("maps/turning_radius/map55.net.xml", "edge-south-NS"),
        ("maps/turning_radius/map78.net.xml", "edge-south-NS"),
        ("maps/turning_radius/map90.net.xml", "edge-south-NS"),
        ("maps/turning_radius/map107.net.xml", "edge-south-NS"),
        ("maps/turning_radius/map128.net.xml", "edge-south-NS"),
    ]
)
def scenarios(request):
    with temp_scenario(name="map", map=request.param[0]) as scenario_root:
        mission = t.Mission(
            route=t.Route(begin=("edge-west-WE", 0, 10), end=(request.param[1], 0, 40))
        )
        gen_scenario(
            t.Scenario(ego_missions=[mission]),
            output_dir=scenario_root,
        )
        yield Scenario.variations_for_all_scenario_roots(
            [str(scenario_root)], [AGENT_ID]
        )


@pytest.fixture
def agent_spec(agent_and_agent_type):
    return AgentSpec(
        interface=AgentInterface.from_type(
            agent_and_agent_type[1], max_episode_steps=5000
        ),
        agent_builder=agent_and_agent_type[0],
    )


@pytest.fixture
def smarts(agent_spec):
    smarts = SMARTS(
        agent_interfaces={AGENT_ID: agent_spec.interface},
        traffic_sim=SumoTrafficSimulation(),
    )
    yield smarts
    smarts.destroy()


def test_lane_following_controller(smarts, agent_spec, scenarios):
    # We introduce a flag `detected` to find the first instance of time for which the
    # speed exceeds 5 km/hr and then we start to record the speed. If after, flag
    # `detected` becomes `True`, any element of speed becomes less than 5 km/hr then
    # the test will fail.
    detected = 0
    lateral_error = []
    speed = []

    agent = agent_spec.build_agent()
    scenario = next(scenarios)
    observations = smarts.reset(scenario)

    agent_obs = None
    for _ in range(500):
        agent_obs = observations[AGENT_ID]
        agent_obs = agent_spec.observation_adapter(agent_obs)

        if agent_obs.ego_vehicle_state.speed > 5 / 3.6:
            detected = 1
        if detected == 1:
            speed.append(agent_obs.ego_vehicle_state.speed)

        current_lane = LaneFollowingController.find_current_lane(
            agent_obs.waypoint_paths, agent_obs.ego_vehicle_state.position
        )
        wp_path = agent_obs.waypoint_paths[current_lane]
        lateral_error.append(
            abs(
                wp_path[0].signed_lateral_error(
                    agent_obs.ego_vehicle_state.position[0:2]
                )
            )
        )

        agent_action = agent_spec.action_adapter(agent.act(agent_obs))
        observations, _, dones, _ = smarts.step({AGENT_ID: agent_action})

        if agent_obs.events.reached_goal:
            break

    assert agent_obs is not None and agent_obs.events.reached_goal, "Didn't reach goal"
    assert min(speed) > 5 / 3.6, "Speed dropped below minimum (5)"
    assert sum(speed) / len(speed) > 5, "Average speed below maximum (5)"
    assert max(lateral_error) < 2.2, "Lateral error exceeded maximum (2)"
    assert (
        sum(lateral_error) / len(lateral_error) < 1
    ), "Average lateral error exceeded maximum (1)"
