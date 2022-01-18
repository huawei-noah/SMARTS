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
from smarts.env.wrappers.standard_obs import StandardObs, StdObs


def _intrfcs():
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
    # fmt: off
    intrfcs = [
        (dataclasses.replace(base_intrfc,accelerometer=True), dataclasses.replace(base_intrfc,accelerometer=False)),
        (dataclasses.replace(base_intrfc,accelerometer=True), dataclasses.replace(base_intrfc,accelerometer=True)),
        (dataclasses.replace(base_intrfc,drivable_area_grid_map=True), dataclasses.replace(base_intrfc,drivable_area_grid_map=True)),
        (dataclasses.replace(base_intrfc,lidar=True), dataclasses.replace(base_intrfc,lidar=True)),
        (dataclasses.replace(base_intrfc,neighborhood_vehicles=True), dataclasses.replace(base_intrfc,neighborhood_vehicles=True)),
        (dataclasses.replace(base_intrfc,ogm=True), dataclasses.replace(base_intrfc,ogm=True)),
        (dataclasses.replace(base_intrfc,rgb=True), dataclasses.replace(base_intrfc,rgb=True)),
        (dataclasses.replace(base_intrfc,waypoints=Waypoints(lookahead=70)), dataclasses.replace(base_intrfc,waypoints=Waypoints(lookahead=70))),
        (dataclasses.replace(base_intrfc,waypoints=Waypoints(lookahead=3)), dataclasses.replace(base_intrfc,waypoints=Waypoints(lookahead=3))),
    ]
    # fmt: on

    return intrfcs


def _make_agent_specs(intrfcs):
    return {
        "AGENT_"
        + agent_id: AgentSpec(
            interface=intrfc,
            agent_builder=lambda: Agent.from_function(lambda _: "keep_lane"),
        )
        for agent_id, intrfc in zip(["001", "002"], intrfcs)
    }


def _make_gym_spec(intrfcs):
    return {
        "AGENT_"
        + agent_id: AgentSpec(
            interface=intrfc,
            agent_builder=lambda: Agent.from_function(lambda _: "keep_lane"),
        )
        for agent_id, intrfc in zip(["001", "002"], intrfcs)
    }


@pytest.fixture
def base_env(request):
    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=["scenarios/figure_eight"],
        agent_specs=_make_agent_specs(request.param),
        headless=True,
        visdom=False,
    )

    yield env
    env.close()


@pytest.mark.parametrize("base_env", _intrfcs(), indirect=True)
def test_init(base_env):
    agent_ids = list(base_env.agent_specs.keys())
    intrfcs = {
        agent_id: base_env.agent_specs[agent_id].interface for agent_id in agent_ids
    }

    # Test wrapping an env with non-identical agent interfaces
    if not all(intrfc == intrfcs[agent_ids[0]] for intrfc in intrfcs.values()):
        with pytest.raises(AssertionError):
            env = StandardObs(env=base_env)
            env.close()
        return
    else:
        env = StandardObs(env=base_env)

    # Test observation space of wrapped env
    obs_space = env.observation_space

    # # Test wrapping an env with and without RGB functionality
    # agent_id = next(iter(base_env.agent_specs.keys()))
    # if base_env.agent_specs[agent_id].interface.rgb == False:
    #     with pytest.raises(AssertionError):
    #         env = RGBImage(base_env, num_stack)
    #         env.close()
    #     return
    # else:
    #     env = RGBImage(base_env, num_stack)

    # # Test wrapped env observation space
    # assert isinstance(env.observation_space, gym.spaces.Dict)
    # for agent_id in base_env.agent_specs.keys():
    #     rgb = base_env.agent_specs[agent_id].interface.rgb
    #     assert env.observation_space[agent_id].shape == (
    #         rgb.width,
    #         rgb.height,
    #         3 * num_stack,
    #     )
    #     assert env.observation_space[agent_id].dtype == np.uint8

    env.close()


# def _check_observation(base_env, obs, num_stack):
#     for agent_id in base_env.agent_specs.keys():
#         rgb = base_env.agent_specs[agent_id].interface.rgb
#         assert obs[agent_id].shape == (rgb.width, rgb.height, 3 * num_stack)
#         assert obs[agent_id].dtype == np.uint8


# @pytest.mark.parametrize("num_stack", [1, 2])
# @pytest.mark.parametrize("base_env", ["rgb"], indirect=True)
# def test_observation(base_env, num_stack):
#     base_env = _frame_stack(base_env, num_stack)
#     env = RGBImage(base_env, num_stack)

#     # Test resetting the env
#     obs = env.reset()
#     _check_observation(base_env, obs, num_stack)

#     # Test stepping the env
#     action = {agent_id: "keep_lane" for agent_id in base_env.agent_specs.keys()}
#     obs, _, _, _ = env.step(action)
#     _check_observation(base_env, obs, num_stack)

#     env.close()
