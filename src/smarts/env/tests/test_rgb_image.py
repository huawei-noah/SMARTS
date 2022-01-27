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

import gym
import numpy as np
import pytest

from smarts.core.agent import Agent, AgentSpec
from smarts.core.agent_interface import RGB, AgentInterface
from smarts.core.controllers import ActionSpaceType
from smarts.env.wrappers.frame_stack import FrameStack
from smarts.env.wrappers.rgb_image import RGBImage


def _make_agent_specs(topdown_rgb):
    if topdown_rgb == "rgb":
        rgb = RGB()
    elif topdown_rgb == "false":
        rgb = False

    return {
        "AGENT_"
        + agent_id: AgentSpec(
            interface=AgentInterface(
                rgb=rgb,
                action=ActionSpaceType.Lane,
            ),
            agent_builder=lambda: Agent.from_function(lambda _: "keep_lane"),
        )
        for agent_id in ["001", "002"]
    }


@pytest.fixture
def base_env(request):
    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=["scenarios/figure_eight"],
        agent_specs=_make_agent_specs(request.param),
        headless=True,
        visdom=False,
        fixed_timestep_sec=0.01,
    )

    yield env
    env.close()


def _frame_stack(env, num_stack):
    if num_stack == 1:
        return env
    return FrameStack(env, num_stack)


@pytest.mark.parametrize("num_stack", [1, 2])
@pytest.mark.parametrize("base_env", ["rgb", "false"], indirect=True)
def test_init(base_env, num_stack):
    base_env = _frame_stack(base_env, num_stack)

    # Test wrapping an env with and without RGB functionality
    agent_id = next(iter(base_env.agent_specs.keys()))
    if base_env.agent_specs[agent_id].interface.rgb == False:
        with pytest.raises(AssertionError):
            env = RGBImage(base_env, num_stack)
            env.close()
        return
    else:
        env = RGBImage(base_env, num_stack)

    # Test wrapped env observation space
    assert isinstance(env.observation_space, gym.spaces.Dict)
    for agent_id in base_env.agent_specs.keys():
        rgb = base_env.agent_specs[agent_id].interface.rgb
        assert env.observation_space[agent_id].shape == (
            rgb.width,
            rgb.height,
            3 * num_stack,
        )
        assert env.observation_space[agent_id].dtype == np.uint8

    env.close()


def _check_observation(base_env, obs, num_stack):
    for agent_id in base_env.agent_specs.keys():
        rgb = base_env.agent_specs[agent_id].interface.rgb
        assert obs[agent_id].shape == (rgb.width, rgb.height, 3 * num_stack)
        assert obs[agent_id].dtype == np.uint8


@pytest.mark.parametrize("num_stack", [1, 2])
@pytest.mark.parametrize("base_env", ["rgb"], indirect=True)
def test_observation(base_env, num_stack):
    base_env = _frame_stack(base_env, num_stack)
    env = RGBImage(base_env, num_stack)

    # Test resetting the env
    obs = env.reset()
    _check_observation(base_env, obs, num_stack)

    # Test stepping the env
    action = {agent_id: "keep_lane" for agent_id in base_env.agent_specs.keys()}
    obs, _, _, _ = env.step(action)
    _check_observation(base_env, obs, num_stack)

    env.close()
