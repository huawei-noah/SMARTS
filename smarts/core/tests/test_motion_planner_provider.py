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
import math

import numpy as np
import pytest

from smarts.core.agent_interface import ActionSpaceType, AgentInterface, DoneCriteria
from smarts.core.agents_provider import MotionPlannerProvider
from smarts.core.coordinates import Heading, Pose
from smarts.core.plan import EndlessGoal, Mission, Start
from smarts.core.provider import ProviderState
from smarts.core.scenario import Scenario
from smarts.core.smarts import SMARTS
from smarts.core.vehicle import ActorRole, VEHICLE_CONFIGS, VehicleState

AGENT_ID = "Agent-007"


@pytest.fixture
def smarts():
    agent = AgentInterface(
        debug=True,
        action=ActionSpaceType.TargetPose,
        done_criteria=DoneCriteria(collision=False, off_road=False, off_route=False),
    )
    agents = {AGENT_ID: agent}
    smarts = SMARTS(agents, fixed_timestep_sec=1.0)
    yield smarts
    smarts.destroy()


@pytest.fixture
def loop_scenario():
    mission = Mission(
        start=Start(np.array((0, 0, 0.5)), Heading(0)),
        goal=EndlessGoal(),
    )
    scenario = Scenario(
        scenario_root="scenarios/sumo/loop",
        traffic_specs=["scenarios/sumo/loop/traffic/basic.rou.xml"],
        missions={AGENT_ID: mission},
    )
    return scenario


def test_we_reach_target_pose_at_given_time(smarts, loop_scenario):
    target_position = [32, -12]
    target_heading = math.pi * 0.5

    observations = smarts.reset(loop_scenario)
    for i in range(10):
        t = 10 - i
        actions = {AGENT_ID: np.array([*target_position, target_heading, t])}
        observations, _, dones, _ = smarts.step(actions)

    ego_state = observations[AGENT_ID].ego_vehicle_state
    assert np.linalg.norm(ego_state.position[:2] - np.array(target_position)) < 1e-16
    assert np.isclose(ego_state.heading, target_heading)
