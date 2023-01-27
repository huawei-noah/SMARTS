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

import dataclasses
from unittest import mock

import gymnasium as gym
import numpy as np
import pytest

from smarts.core.agent_interface import AgentInterface, DoneCriteria
from smarts.core.controllers import ActionSpaceType
from smarts.core.coordinates import Heading, Point
from smarts.core.plan import EndlessGoal, Goal, Mission, PositionalGoal, Start
from smarts.env.gymnasium.wrappers.metrics import Metrics
from smarts.zoo.agent_spec import AgentSpec


def _intrfc_improper():
    return [
        {"accelerometer": False},
        {"max_episode_steps": None},
        {"neighborhood_vehicle_states": False},
        {"road_waypoints": False},
        {"waypoint_paths": False},
        {
            "done_criteria": DoneCriteria(
                collision=False,
                off_road=True,
            )
        },
        {
            "done_criteria": DoneCriteria(
                collision=True,
                off_road=False,
            )
        },
    ]


@pytest.fixture
def get_agent_spec(request):
    base_intrfc = AgentInterface(
        action=ActionSpaceType.TargetPose,
        accelerometer=True,
        done_criteria=DoneCriteria(
            collision=True,
            off_road=True,
            off_route=False,
            on_shoulder=False,
            wrong_way=False,
            not_moving=False,
            agents_alive=None,
        ),
        max_episode_steps=5,
        neighborhood_vehicle_states=True,
        waypoint_paths=True,
        road_waypoints=True,
    )
    return AgentSpec(interface=dataclasses.replace(base_intrfc, **request.param))


@pytest.fixture(scope="module")
def get_scenario(request):
    from pathlib import Path

    from smarts.sstudio.scenario_construction import build_scenario

    if request.param == "single_agent_intersection":
        scenario = str(
            Path(__file__).resolve().parents[2]
            / "scenarios"
            / "intersection"
            / "1_to_1lane_left_turn_c"
        )
        num_agents = 1
    elif request.param == "multi_agent_merge":
        scenario = str(
            Path(__file__).resolve().parents[2]
            / "scenarios"
            / "merge"
            / "3lane_multi_agent"
        )
        num_agents = 2

    build_scenario(scenario=scenario)

    return (scenario, num_agents)


@pytest.fixture
def make_env(get_agent_spec, get_scenario):
    env = gym.make(
        "smarts.env:hiway-v1",
        scenarios=[get_scenario[0]],
        agent_interfaces={
            f"AGENT_{agent_id}": get_agent_spec.interface
            for agent_id in range(get_scenario[1])
        },
        headless=True,
    )
    yield env
    env.close()


@pytest.mark.parametrize("get_agent_spec", _intrfc_improper(), indirect=True)
@pytest.mark.parametrize("get_scenario", ["single_agent_intersection"], indirect=True)
def test_improper_interface(make_env):

    # Verify proper agent interface enabled.
    with pytest.raises(AttributeError):
        env = Metrics(env=make_env)


@pytest.mark.parametrize("get_agent_spec", [{}], indirect=True)
@pytest.mark.parametrize("get_scenario", ["single_agent_intersection"], indirect=True)
def test_init(make_env):

    # Verify instantiation of Metrics wrapper.
    env = Metrics(env=make_env)

    # Verify blocked access to underlying private variables.
    for elem in ["_scen", "_road_map", "_records"]:
        with pytest.raises(AttributeError):
            getattr(env, elem)


def _mock_mission(start: Start, goal: Goal):
    def func(scenario_root, agents_to_be_briefed):
        return [Mission(start=start, goal=goal)]

    return func


@pytest.mark.parametrize("get_agent_spec", [{}], indirect=True)
@pytest.mark.parametrize("get_scenario", ["single_agent_intersection"], indirect=True)
def test_reset(make_env):

    # Verify a scenario without PositionalGoal fails suitability check.
    with mock.patch(
        "smarts.core.scenario.Scenario.discover_agent_missions",
        side_effect=_mock_mission(
            start=Start(position=np.array([0, 0, 0]), heading=Heading(0)),
            goal=EndlessGoal(),
        ),
    ):
        with pytest.raises(AttributeError):
            env = Metrics(env=make_env)
            env.reset()
        return


@pytest.mark.parametrize("get_agent_spec", [{}], indirect=True)
@pytest.mark.parametrize("get_scenario", ["single_agent_intersection"], indirect=True)
def test_end_in_off_road(make_env):

    # Verify that env.score() can be computed when vehicle goes off road.
    env = Metrics(env=make_env)
    obs, _ = env.reset()
    agent_name = next(iter(env.agent_interfaces.keys()))
    dones = {"__all__": False}
    while not dones["__all__"]:
        actions = {
            agent_name: np.append(
                obs[agent_name]["ego_vehicle_state"]["position"][:2]
                + np.array([0.5, -0.8]),
                [obs[agent_name]["ego_vehicle_state"]["heading"], 0.1],
            )
        }
        obs, _, dones, _, _ = env.step(actions)
    assert obs[agent_name]["events"]["off_road"], (
        "Expected vehicle to go off road, but it did not. "
        f"Events: {obs[agent_name]['events']}."
    )
    env.score()

    # Verify that Count values increase with episode.
    records = env.records()
    scen_name = next(iter(records.keys()))
    counts = records[scen_name][agent_name].counts
    assert counts.goals == 0
    assert counts.episodes == 1
    assert counts.steps == 3
    assert counts.max_steps == env.agent_interfaces[agent_name].max_episode_steps


@pytest.mark.parametrize(
    "get_agent_spec",
    [{"max_episode_steps": 27, "done_criteria": DoneCriteria(off_route=True)}],
    indirect=True,
)
@pytest.mark.parametrize("get_scenario", ["single_agent_intersection"], indirect=True)
def test_end_in_off_route(make_env):

    # Verify that env.score() can be computed when vehicle ends in off route.
    # Note:
    #   Point(-12, -1.6, 0) lies on edge-west-WE_0, i.e., to the left of the junction.
    #   Point( 12, -1.6, 0) lies on edge-east-WE_0, i.e., to the right of the junction.
    #   Point(1.5, 30.5, 0) lies on edge-north-SN_0, i.e., to the top of the junction.
    with mock.patch(
        "smarts.core.scenario.Scenario.discover_agent_missions",
        side_effect=_mock_mission(
            start=Start(position=np.array([-12, -1.6, 0]), heading=Heading(-1.57)),
            goal=PositionalGoal(position=Point(x=1.5, y=30.5, z=0), radius=3),
        ),
    ):
        env = Metrics(env=make_env)
        obs, _ = env.reset()
        agent_name = next(iter(env.agent_interfaces.keys()))
        dones = {"__all__": False}
        while not dones["__all__"]:
            actions = {
                agent_name: np.append(
                    obs[agent_name]["ego_vehicle_state"]["position"][:2]
                    + np.array([1, 0]),
                    [obs[agent_name]["ego_vehicle_state"]["heading"], 0.1],
                )
            }
            obs, _, dones, _, _ = env.step(actions)
        assert (
            obs[agent_name]["ego_vehicle_state"]["lane_id"].rstrip() == "edge-east-WE_0"
        ), (
            "Expected vehicle to drive off route, but it is at lane: "
            f"{obs[agent_name]['ego_vehicle_state']['lane_id']}."
        )
        assert obs[agent_name]["events"]["off_route"], (
            "Expected vehicle to go off route, but it did not. "
            f"Events: {obs[agent_name]['events']}."
        )
        env.score()


@pytest.mark.parametrize("get_agent_spec", [{"max_episode_steps": 1}], indirect=True)
@pytest.mark.parametrize("get_scenario", ["single_agent_intersection"], indirect=True)
def test_end_in_junction(make_env):

    # Verify that env.score() can be computed when vehicle ends in a junction.
    # Note:
    #   Point(-1.76, 2.05, 0) lies on :junction-intersection_1_0, i.e., inside the junction.
    with mock.patch(
        "smarts.core.scenario.Scenario.discover_agent_missions",
        side_effect=_mock_mission(
            start=Start(position=np.array([-1.86, 1.95, 0]), heading=Heading(-1.00)),
            goal=PositionalGoal(position=Point(x=1.5, y=30.5, z=0), radius=3),
        ),
    ):
        env = Metrics(env=make_env)
        obs, _ = env.reset()
        agent_name = next(iter(obs.keys()))
        actions = {
            agent_id: np.array([-1.76, 2.05, -0.91, 0.1]) for agent_id in obs.keys()
        }
        obs, _, dones, _, _ = env.step(actions)
        assert (
            obs[agent_name]["ego_vehicle_state"]["lane_id"].rstrip()
            == ":junction-intersection_1_0"
        ), (
            "Expected vehicle to be inside junction, but it is at lane: "
            f"{obs[agent_name]['ego_vehicle_state']['lane_id']}."
        )
        assert (
            obs[agent_name]["events"]["reached_max_episode_steps"] and dones["__all__"]
        ), (
            "Expected vehicle to reach max episode steps and become done, but "
            f"it did not. Dones: {dones}. Events: {obs[agent_name]['events']}."
        )
        env.score()


@pytest.mark.parametrize("get_agent_spec", [{}], indirect=True)
@pytest.mark.parametrize("get_scenario", ["multi_agent_merge"], indirect=True)
def test_records_and_scores(make_env):

    # Verify that records and scores are functional in multi-agent environment.
    # Note:
    #   env.score() is only callable after >=1 episode. Hence step through 1 episode.
    env = Metrics(env=make_env)
    obs, _ = env.reset()
    agent_name = next(iter(obs.keys()))
    for _ in range(env.agent_interfaces[agent_name].max_episode_steps):
        actions = {
            agent_name: np.append(
                agent_obs["ego_vehicle_state"]["position"][:2], [0, 0.1]
            )
            for agent_name, agent_obs in obs.items()
        }
        obs, _, _, _, _ = env.step(actions)
    env.records()
    env.score()
