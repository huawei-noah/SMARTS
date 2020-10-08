import math
import multiprocessing
from itertools import cycle

import pytest

from smarts.core.agent_interface import AgentInterface, ActionSpaceType
from smarts.core.scenario import Scenario, Mission, Start, EndlessGoal
from smarts.core.smarts import SMARTS
from smarts.core.sumo_traffic_simulation import SumoTrafficSimulation
from smarts.core.coordinates import Heading
from smarts.core.utils.sumo import traci

SUMO_PORT = 8082


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
def traffic_sim():
    return SumoTrafficSimulation(
        headless=True, num_external_sumo_clients=1, sumo_port=SUMO_PORT
    )


@pytest.fixture
def smarts(traffic_sim):
    buddha = AgentInterface(
        max_episode_steps=1000, neighborhood_vehicles=True, action=ActionSpaceType.Lane,
    )
    agents = {"Agent-007": buddha}
    smarts = SMARTS(agents, traffic_sim=traffic_sim)

    yield smarts
    smarts.destroy()


def connection():
    """This connects to an existing SUMO server instance."""
    traci_conn = traci.connect(
        SUMO_PORT, numRetries=100, proc=None, waitBetweenRetries=0.1
    )
    traci_conn.setOrder(2)
    return traci_conn


def run_client():
    conn = connection()
    for _ in range(10):
        conn.simulationStep()

    conn.close()


def run_smarts(smarts, scenarios):
    scenario = next(scenarios)
    smarts.reset(scenario)
    for _ in range(10):
        smarts.step({"Agent-007": "keep_lane"})


def test_traffic_sim_with_multi_client(smarts, scenarios):
    pool = multiprocessing.Pool()
    pool.apply_async(run_smarts, args=(smarts, scenarios))
    pool.apply_async(run_client, args=())

    pool.close()
    pool.join()
