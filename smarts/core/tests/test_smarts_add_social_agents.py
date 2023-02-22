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
from itertools import cycle
from typing import Iterator

import numpy as np
import pytest

from smarts.core.agent_interface import (
    ActionSpaceType,
    AgentInterface,
    DoneCriteria,
    NeighborhoodVehicles,
)
from smarts.core.coordinates import Heading
from smarts.core.plan import EndlessGoal, Mission, Start
from smarts.core.scenario import Scenario
from smarts.core.smarts import SMARTS
from smarts.core.sumo_traffic_simulation import SumoTrafficSimulation
from smarts.core.utils.custom_exceptions import RendererException
from smarts.core.provider import Provider


@pytest.fixture
def scenario():
    mission = Mission(
        start=Start(np.array((71.65, 63.78)), Heading(math.pi * 0.91)),
        goal=EndlessGoal(),
    )
    scenario = Scenario(
        scenario_root="scenarios/sumo/loop",
        traffic_specs=["scenarios/sumo/loop/build/traffic/basic.rou.xml"],
        missions={"Agent-007": mission},
    )
    return scenario


@pytest.fixture
def smarts():
    buddha = AgentInterface(
        max_episode_steps=1000,
        neighborhood_vehicle_states=NeighborhoodVehicles(radius=20),
        action=ActionSpaceType.Lane,
        done_criteria=DoneCriteria(collision=False, off_road=False),
    )
    agents = {"Agent-007": buddha}
    smarts = SMARTS(
        agents,
        traffic_sims=[SumoTrafficSimulation(headless=False, auto_start=False)],
        envision=None,
    )

    yield smarts
    smarts.destroy()


def test_smarts_control_actors_with_social_agents(smarts: SMARTS, scenario: Scenario):
    smarts.reset(scenario)

    for _ in range(10):
        smarts.step({})

    provider: Provider = smarts.get_provider_by_type(SumoTrafficSimulation)
    assert provider.actor_ids
    vehicle_ids = set(list(provider.actor_ids)[:5] + ["Agent-007"])
    original_social_agent_vehicle_ids = smarts.vehicle_index.agent_vehicle_ids()
    agent_ids, rejected_vehicle_ids = smarts.control_actors_with_social_agents(
        "scenarios.sumo.zoo_intersection.agent_prefabs:zoo-agent2-v0", vehicle_ids
    )

    new_social_agent_vehicle_ids = smarts.vehicle_index.agent_vehicle_ids()
    assert rejected_vehicle_ids
    assert smarts.agent_manager.social_agent_ids.issuperset(agent_ids)
    assert agent_ids
    assert new_social_agent_vehicle_ids.isdisjoint(rejected_vehicle_ids)
    assert len(original_social_agent_vehicle_ids) < len(new_social_agent_vehicle_ids)

    for _ in range(10):
        smarts.step({})
