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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import dataclasses

import gym
import numpy as np
import pytest

from smarts.core.agent import Agent, AgentSpec
from smarts.core.agent_interface import AgentInterface, Waypoints
from smarts.core.controllers import ActionSpaceType
from smarts.env.wrappers.standard_obs import (
    StandardObs,
    StdObs,
    get_spaces,
    intrfc_to_stdobs,
)


def _intrfcs_init():
    # fmt: off
    intrfcs = [
        ({"accelerometer":True}, {"accelerometer":False}),
        ({"drivable_area_grid_map":True}, {"drivable_area_grid_map":True}),
        ({"lidar":True}, {"lidar":True}),
        ({"neighborhood_vehicles":True}, {"neighborhood_vehicles":True}),
        ({"ogm":True}, {"ogm":True}),
        ({"rgb":True}, {"rgb":True}),
        ({"waypoints":Waypoints(lookahead=1)}, {"waypoints":Waypoints(lookahead=1)}),
        ({"neighborhood_vehicles":True, "waypoints":Waypoints(lookahead=1)}, {"neighborhood_vehicles":True, "waypoints":Waypoints(lookahead=1)}),
    ]
    # fmt: on

    return intrfcs


def _intrfcs_obs():
    # fmt: off
    intrfcs = [
        ({
            "accelerometer":True, 
            "drivable_area_grid_map":True, 
            "lidar":True,
            "neighborhood_vehicles":True, 
            "ogm":True, 
            "rgb":True, 
            "waypoints":Waypoints(lookahead=60),
        }, 
        {   
            "accelerometer":True,
            "drivable_area_grid_map":True,
            "lidar":True,
            "neighborhood_vehicles":True,
            "ogm":True,
            "rgb":True,
            "waypoints":Waypoints(lookahead=60),
        }),
        ({
            "accelerometer":True,
            "drivable_area_grid_map":True,
            "lidar":True,
            "neighborhood_vehicles":True,
            "ogm":True,
            "rgb":True,
            "waypoints":Waypoints(lookahead=1),
        }, 
        {"accelerometer":True,
        "drivable_area_grid_map":True,
        "lidar":True,
        "neighborhood_vehicles":True,
        "ogm":True,
        "rgb":True,
        "waypoints":Waypoints(lookahead=1),
        }),    ]
    # fmt: on

    return intrfcs


def _make_agent_specs(intrfcs):
    base_intrfc = AgentInterface(
        action=ActionSpaceType.Lane,
        accelerometer=False,
        drivable_area_grid_map=False,
        lidar=False,
        neighborhood_vehicles=False,
        ogm=False,
        rgb=False,
        waypoints=False,
    )

    return {
        "AGENT_"
        + agent_id: AgentSpec(
            interface=dataclasses.replace(base_intrfc, **intrfc),
            agent_builder=lambda: Agent.from_function(lambda _: "keep_lane"),
        )
        for agent_id, intrfc in zip(["001", "002"], intrfcs)
    }


@pytest.fixture
def make_env(request):
    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=["scenarios/figure_eight"],
        agent_specs=_make_agent_specs(request.param),
        headless=True,
        visdom=False,
    )

    yield env, request.param
    env.close()


@pytest.fixture(scope="module")
def spaces():
    return get_spaces()


@pytest.mark.parametrize("make_env", _intrfcs_init(), indirect=True)
def test_init(make_env, spaces):
    base_env, cur_intrfcs = make_env

    # Test wrapping an env with non-identical agent interfaces
    base_intrfcs = [
        agent_spec.interface for agent_spec in base_env.agent_specs.values()
    ]
    if not all(intrfc == base_intrfcs[0] for intrfc in base_intrfcs):
        with pytest.raises(AssertionError):
            env = StandardObs(env=base_env)
            env.close()
        return
    else:
        env = StandardObs(env=base_env)

    # Test observation space of wrapped env
    agent_id = next(iter(base_env.agent_specs.keys()))
    rcv_space = env.observation_space
    rcv_space_keys = set([key for key in rcv_space[agent_id]])

    basic, _ = spaces
    des_space_keys = set(basic.keys())
    opt_space_keys = [
        intrfc_to_stdobs(intrfc)
        for intrfc, val in cur_intrfcs[0].items()
        if val and intrfc_to_stdobs(intrfc)
    ]
    des_space_keys.update(opt_space_keys)
    if "waypoints" in opt_space_keys and "neighbors" in opt_space_keys:
        des_space_keys.update(["ttc"])
    assert rcv_space_keys == des_space_keys

    env.close()


# def _check_observation(
#     base_env,
#     obs,
# ):
#     for agent_id in base_env.agent_specs.keys():
#         rgb = base_env.agent_specs[agent_id].interface.rgb
#         assert obs[agent_id].shape == (rgb.width, rgb.height, 3 * num_stack)
#         assert obs[agent_id].dtype == np.uint8


# @pytest.mark.parametrize("make_env", _intrfcs(), indirect=True)
# def test_observation(make_env, num_stack):
#     base_env, cur_intrfcs = make_env

#     env = StandardObs(env=base_env)
#     rcv_space = env.observation_space

#     obs = env.reset()

#     # Test resetting the env
#     obs = env.reset()
#     _check_observation(base_env, obs, num_stack)

#     # Test stepping the env
#     action = {agent_id: "keep_lane" for agent_id in base_env.agent_specs.keys()}
#     obs, _, _, _ = env.step(action)
#     _check_observation(base_env, obs, num_stack)

#     env.close()
