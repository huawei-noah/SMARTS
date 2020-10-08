import math

from itertools import cycle

import pytest
import numpy as np

from smarts.core.smarts import SMARTS
from smarts.core.sumo_traffic_simulation import SumoTrafficSimulation
from smarts.core.scenario import Scenario, Mission, Start, EndlessGoal
from smarts.core.coordinates import Heading
from smarts.core.agent_interface import (
    AgentInterface,
    ActionSpaceType,
    NeighborhoodVehicles,
)


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
        agents, traffic_sim=SumoTrafficSimulation(headless=True), envision=None,
    )

    yield smarts
    smarts.destroy()


def test_smarts_doesnt_leak_tasks_after_reset(smarts, scenarios):
    """We have had issues in the past where we would forget to clean up tasks between episodes
    resulting in a gradual decay in performance, this test gives us a bit of a smoke screen
    against this class of regressions.

    See #237 for details
    """
    num_tasks_before_reset = len(smarts.taskMgr.mgr.getTasks())

    scenario = next(scenarios)
    smarts.reset(scenario)

    for _ in range(10):
        smarts.step({})

    num_tasks_after_reset = len(smarts.taskMgr.mgr.getTasks())
    assert num_tasks_after_reset == num_tasks_before_reset
