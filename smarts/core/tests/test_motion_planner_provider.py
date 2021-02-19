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
import math

import numpy as np
import pytest

from smarts.core.coordinates import Heading, Pose
from smarts.core.motion_planner_provider import MotionPlannerProvider
from smarts.core.provider import ProviderState
from smarts.core.scenario import EndlessGoal, Mission, Scenario, Start
from smarts.core.vehicle import VEHICLE_CONFIGS, VehicleState


@pytest.fixture
def motion_planner_provider():
    return MotionPlannerProvider()


@pytest.fixture
def loop_scenario():
    mission = Mission(
        start=Start((71.65, 63.78), Heading(math.pi * 0.91)), goal=EndlessGoal()
    )
    scenario = Scenario(
        scenario_root="scenarios/loop",
        route="basic.rou.xml",
        missions={"Agent-007": mission},
    )
    return scenario


def test_we_reach_target_pose_at_given_time(motion_planner_provider, loop_scenario):
    motion_planner_provider.setup(loop_scenario)

    # we sync with the empty provider state since we don't have any other active providers
    motion_planner_provider.sync(ProviderState())

    motion_planner_provider.create_vehicle(
        VehicleState(
            vehicle_id="EGO",
            vehicle_type="passenger",
            pose=Pose.from_center([0, 0, 0.5], heading=Heading(0)),
            dimensions=VEHICLE_CONFIGS["passenger"].dimensions,
            speed=0,
            source="TESTS",
        )
    )
    target_position = [32, -12]
    target_heading = math.pi * 0.5
    dt = 1.0
    elapsed_sim_time = 0
    for i in range(10):
        t = 10 - i
        provider_state = motion_planner_provider.step(
            dt=dt,
            target_poses_at_t={"EGO": np.array([*target_position, target_heading, t])},
            elapsed_sim_time=elapsed_sim_time,
        )
        elapsed_sim_time += dt

    assert len(provider_state.vehicles) == 1
    ego_vehicle = provider_state.vehicles[0]
    position, heading = ego_vehicle.pose.position, ego_vehicle.pose.heading

    assert np.linalg.norm(position[:2] - np.array(target_position)) < 1e-16
    assert np.isclose(heading, target_heading)
