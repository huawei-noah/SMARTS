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

import gym
import pytest
import dataclasses
from smarts.core.agent_interface import AgentInterface
from smarts.core.controllers import ActionSpaceType
from smarts.zoo.agent_spec import AgentSpec
from smarts.env.wrappers.metrics import Metrics
from smarts.core.agent_interface import (
    AgentInterface,
    DoneCriteria,
)
from smarts.core.plan import EndlessGoal

from typing import Dict

def _intrfc_improper():
    return [
        {"accelerometer":False},
        {"max_episode_steps":None},
        {"neighborhood_vehicles":False},
        {"waypoints":False},
        {"road_waypoints":False},
        {"done_criteria":DoneCriteria(
            collision=False,
            off_road=True,
        )},
        {"done_criteria":DoneCriteria(
            collision=True,
            off_road=False,
        )}
    ]

def make_agent_specs(intrfc:Dict):
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
    return {
        "AGENT_"
        + agent_id: AgentSpec(
            interface=dataclasses.replace(base_intrfc, **intrfc),
        )
        for agent_id in ["001", "002"]
    }

@pytest.fixture
def make_env(request):
    print(request.param)

    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=["smarts/scenarios/intersection/1_to_1lane_left_turn_c"],
        agent_specs=make_agent_specs(request.param),
        headless=True,
        sumo_headless=True,
        visdom=False,
        fixed_timestep_sec=0.01,
    )

    yield env
    env.close()

@pytest.mark.parametrize("make_env", [{}]+_intrfc_improper(), indirect=True, ids=["Proper"]+["Improper"]*7)


def test_init(request, make_env):
    # Verify proper agent interface enabled
    param_id = request.node.callspec.id
    if param_id == "Proper":
        env = Metrics(env=make_env)
    else:
        with pytest.raises(AttributeError):
            env = Metrics(env=make_env)
        return

    # Verify blocked access to underlying private variables
    for elem in ["_scen", "_road_map", "_records"]:
        with pytest.raises(AttributeError):
            getattr(env, elem)
        
# @pytest.mark.parametrize("make_env", [{}], indirect=True)
# def test_reset(make_env):
#     env = make_env
#     env.reset()

    # verify scenario without positional goal
    # print(env.scenario.missions)
    # for _, agent_mission in env.scenario.missions.items():
    #     agent_mission.goal = EndlessGoal()

    # with pytest.raises(AttributeError):
    #     env = Metrics(env=env)
    #     env.reset()
    #     env.close()
    # return


# @pytest.mark.parametrize("make_env", {}, indirect=True)
# def test_step(make_env):
#     # verify whether count values changed: the step, episodes,max-Steps counts have increaded
#     env = make_env

#     env.reset


# def test_score()
#     # verify wheterh tthe score has 

# def test_records):
#     # verify wheterh tthe records contains dictionary  
#     # access private underlying variables


# @pytest.mark.parametrize("num_stack", [1, 2])
# def test_frame_stack(env, agent_specs, num_stack):
#     # Test invalid num_stack inputs
#     if num_stack <= 1:
#         with pytest.raises(Exception):
#             env = FrameStack(env, num_stack)
#         return

#     # Wrap env with FrameStack to stack multiple observations
#     env = FrameStack(env, num_stack)
#     agents = {
#         agent_id: agent_spec.build_agent()
#         for agent_id, agent_spec in agent_specs.items()
#     }

#     # Test whether env.reset returns stacked duplicated observations
#     obs = env.reset()
#     assert len(obs) == len(agents)
#     for agent_id, agent_obs in obs.items():
#         rgb = agent_specs[agent_id].interface.rgb
#         agent_obs = np.asarray(agent_obs)
#         assert agent_obs.shape == (num_stack, rgb.width, rgb.height, 3)
#         for i in range(1, num_stack):
#             assert np.allclose(agent_obs[i - 1], agent_obs[i])

#     # Test whether env.step removes old and appends new observation
#     actions = {
#         agent_id: agents[agent_id].act(agent_obs) for agent_id, agent_obs in obs.items()
#     }
#     obs, _, _, _ = env.step(actions)
#     assert len(obs) == len(agents)
#     for agent_id, agent_obs in obs.items():
#         rgb = agent_specs[agent_id].interface.rgb
#         agent_obs = np.asarray(agent_obs)
#         assert agent_obs.shape == (num_stack, rgb.width, rgb.height, 3)
#         for i in range(1, num_stack - 1):
#             assert np.allclose(agent_obs[i - 1], agent_obs[i])
#         if num_stack > 1:
#             assert not np.allclose(agent_obs[-1], agent_obs[-2])

#     env.close()
