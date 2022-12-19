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

import gym
import numpy as np
import pytest

from smarts.core.agent_interface import AgentInterface, DoneCriteria
from smarts.core.controllers import ActionSpaceType
from smarts.core.coordinates import Heading, Point
from smarts.core.plan import EndlessGoal, Goal, Mission, PositionalGoal, Start
from smarts.env.wrappers.metrics import Metrics
from smarts.zoo.agent_spec import AgentSpec


def _intrfc_improper():
    return [
        {"accelerometer": False},
        {"max_episode_steps": None},
        {"neighborhood_vehicles": False},
        {"road_waypoints": False},
        {"waypoints": False},
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
        neighborhood_vehicles=True,
        waypoints=True,
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

    build_scenario(clean=False, scenario=scenario, seed=42)

    return (scenario, num_agents)


@pytest.fixture
def make_env(get_agent_spec, get_scenario):
    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=[get_scenario[0]],
        agent_specs={
            f"AGENT_{agent_id}": get_agent_spec for agent_id in range(get_scenario[1])
        },
        headless=True,
        sumo_headless=True,
    )
    yield env
    env.close()


@pytest.mark.parametrize(
    "get_agent_spec",
    [{}] + _intrfc_improper(),
    indirect=True,
    ids=["properIntrfc"] + ["improperIntrfc"] * 7,
)
@pytest.mark.parametrize(
    "get_scenario",
    ["single_agent_intersection"],
    indirect=True,
    ids=["intersection"],
)
def test_init(request, make_env):

    # Verify proper agent interface enabled.
    param_id = request.node.callspec.id
    if param_id == "intersection-properIntrfc":
        env = Metrics(env=make_env)
    else:
        with pytest.raises(AttributeError):
            env = Metrics(env=make_env)
        return

    # Verify blocked access to underlying private variables.
    for elem in ["_scen", "_road_map", "_records"]:
        with pytest.raises(AttributeError):
            getattr(env, elem)


def mock_mission(start: Start, goal: Goal):
    def func(scenario_root, agents_to_be_briefed):
        return [Mission(start=start, goal=goal)]

    return func


@pytest.mark.parametrize("get_agent_spec", [{}], indirect=True)
@pytest.mark.parametrize("get_scenario", ["single_agent_intersection"], indirect=True)
def test_reset(make_env):

    # Verify a scenario without PositionalGoal fails suitability check.
    with mock.patch(
        "smarts.core.scenario.Scenario.discover_agent_missions",
        side_effect=mock_mission(
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
def test_step_off_road(make_env):

    # Verify that env.score() can be computed when vehicle goes off road.
    env = Metrics(env=make_env)
    obs = env.reset()
    agent_name = next(iter(obs.keys()))
    max_episode_steps = env.agent_specs[agent_name].interface.max_episode_steps
    for _ in range(max_episode_steps):
        actions = {
            agent_name: np.append(
                agent_obs.ego_vehicle_state.position[:2] + np.array([0.5, -0.8]),
                [agent_obs.ego_vehicle_state.heading, 0.1],
            )
            for agent_name, agent_obs in obs.items()
        }
        obs, _, dones, _ = env.step(actions)
        if dones["__all__"]:
            break
    assert obs[agent_name].events.off_road, (
        "Expected vehicle to go off road, but it did not. "
        f"Events {obs[agent_name].events}."
    )
    env.score()

    # Verify that Count values increase with episode.
    records = env.records()
    scen_name = next(iter(records.keys()))
    counts = records[scen_name][agent_name].counts
    assert counts.goals == 0
    assert counts.episodes == 1
    assert counts.steps == 3
    assert counts.max_steps == max_episode_steps


@pytest.mark.parametrize("get_agent_spec", [{}], indirect=True)
@pytest.mark.parametrize("get_scenario", ["multi_agent_merge"], indirect=True)
def test_records_scores(make_env):

    # Verify that records and scores are accessible and functional.
    # env.score() is only callable after >=1 episode. Hence step through 1 episode.
    env = Metrics(env=make_env)
    obs = env.reset()
    agent_name = next(iter(obs.keys()))
    for _ in range(env.agent_specs[agent_name].interface.max_episode_steps):
        actions = {
            agent_name: np.append(agent_obs.ego_vehicle_state.position[:2], [0, 0.1])
            for agent_name, agent_obs in obs.items()
        }
        obs, _, _, _ = env.step(actions)
    env.records()
    env.score()
