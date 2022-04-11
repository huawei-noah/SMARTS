from typing import Iterator

import pytest

import smarts.sstudio.types as t
from smarts.core.agent import Agent
from smarts.core.agent_interface import AgentInterface
from smarts.core.controllers import ActionSpaceType
from smarts.core.scenario import Scenario
from smarts.core.smarts import SMARTS
from smarts.core.sumo_traffic_simulation import SumoTrafficSimulation
from smarts.core.tests.helpers.scenario import temp_scenario
from smarts.core.utils.adapters.ego_centric_observation_adapter import ego_centric_observation_adapter
from smarts.sstudio.genscenario import gen_scenario
from smarts.zoo.agent_spec import AgentSpec

AGENT_ID = "AGENT-007"


@pytest.fixture
def agent_spec():
    return AgentSpec(
        interface=AgentInterface(
            drivable_area_grid_map=True,
            ogm=True,
            rgb=True,
            lidar=True,
            waypoints=True,
            max_episode_steps=10,
            debug=True,
            neighborhood_vehicles=True,
            action=ActionSpaceType.Lane,
            road_waypoints=True,
        ),
        observation_adapter=ego_centric_observation_adapter,
        agent_builder=lambda: Agent.from_function(lambda _: "keep_lane"),
    )


@pytest.fixture
def bubble():
    return t.Bubble(
        zone=t.PositionalZone(pos=(100, 0), size=(20, 20)),
        margin=10,
        actor=t.BoidAgentActor(
            # TODO: Provide a more self-contained way to build agent locators for tests
            name="hive-mind",
            agent_locator="scenarios.straight.agent_prefabs:pose-boid-agent-v0",
        ),
    )


@pytest.fixture
def scenarios(bubble):
    with temp_scenario(name="straight", map="maps/straight.net.xml") as scenario_root:
        traffic = t.Traffic(
            flows=[
                t.Flow(
                    route=t.Route(
                        begin=("west", lane_idx, 0),
                        end=("east", lane_idx, "max"),
                    ),
                    rate=50,
                    actors={
                        t.TrafficActor("car"): 1,
                    },
                )
                for lane_idx in range(3)
            ]
        )

        gen_scenario(
            t.Scenario(traffic={"all": traffic}, bubbles=[bubble]),
            output_dir=scenario_root,
        )

        yield Scenario.variations_for_all_scenario_roots(
            [str(scenario_root)], [AGENT_ID]
        )


@pytest.fixture
def smarts(agent_spec: AgentSpec):
    smarts = SMARTS(
        {AGENT_ID: agent_spec.interface}, traffic_sim=SumoTrafficSimulation()
    )
    yield smarts
    smarts.destroy()


def test_ego_centric_observation_adapter(smarts: SMARTS, scenarios: Iterator[Scenario], agent_spec: AgentSpec):
    scenario = next(scenarios)
    obs = smarts.reset(scenario)

    new_obs = ego_centric_observation_adapter(obs[AGENT_ID])

    assert obs != new_obs
