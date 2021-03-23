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
from itertools import cycle

import numpy as np
import pytest

from smarts.core.agent_interface import (
    ActionSpaceType,
    AgentInterface,
    NeighborhoodVehicles,
)
from smarts.core.coordinates import Heading
from smarts.core.scenario import EndlessGoal, Mission, Scenario, Start
from smarts.core.smarts import SMARTS
from smarts.core.sumo_traffic_simulation import SumoTrafficSimulation


@pytest.fixture
def scenarios():
    mission = Mission(
        start=Start((71.65, 63.78), Heading(math.pi * 0.91)), goal=EndlessGoal()
    )
    scenario = Scenario(
        scenario_root="scenarios/loop",
        route="basic.rou.xml",
        missions={"Agent-007": mission},
    )
    return cycle([scenario])


@pytest.fixture
def smarts():
    buddha = AgentInterface(
        max_episode_steps=1000,
        neighborhood_vehicles=NeighborhoodVehicles(radius=20),
        action=ActionSpaceType.Lane,
    )
    agents = {"Agent-007": buddha}
    smarts = SMARTS(
        agents,
        traffic_sim=SumoTrafficSimulation(headless=True),
        envision=None,
    )

    yield smarts
    smarts.destroy()


def test_smarts_doesnt_leak_tasks_after_reset(smarts, scenarios):
    """We have had issues in the past where we would forget to clean up tasks between episodes
    resulting in a gradual decay in performance, this test gives us a bit of a smoke screen
    against this class of regressions.

    See #237 for details
    """
    num_tasks_before_reset = len(
        smarts.renderer._showbase_instance.taskMgr.mgr.getTasks()
    )

    scenario = next(scenarios)
    smarts.reset(scenario)

    for _ in range(10):
        smarts.step({})

    num_tasks_after_reset = len(
        smarts.renderer._showbase_instance.taskMgr.mgr.getTasks()
    )
    assert num_tasks_after_reset == num_tasks_before_reset
