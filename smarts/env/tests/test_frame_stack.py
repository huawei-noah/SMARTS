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
from smarts.core.agent_interface import AgentInterface, RGB
from smarts.core.controllers import ActionSpaceType
from smarts.env.wrappers.frame_stack import FrameStack


@pytest.fixture
def agent_specs():
    def observation_adapter(env_observation):
        return env_observation.top_down_rgb.data

    return {
        "AGENT_"
        + agent_id: AgentSpec(
            interface=AgentInterface(
                rgb=RGB(),
                action=ActionSpaceType.Lane,
            ),
            agent_builder=lambda: Agent.from_function(lambda _: "keep_lane"),
            observation_adapter=observation_adapter,
        )
        for agent_id in ["001", "002"]
    }


@pytest.fixture
def env(agent_specs):
    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=["scenarios/loop"],
        agent_specs=agent_specs,
        headless=True,
        visdom=False,
        timestep_sec=0.01,
    )

    yield env
    env.close()


@pytest.mark.parametrize("num_stack", [1, 2])
def test_frame_stack(env, agent_specs, num_stack):
    # Test invalid num_stack inputs
    if num_stack <= 1:
        with pytest.raises(Exception):
            env = FrameStack(env, num_stack)
        return

    # Wrap env with FrameStack to stack multiple observations
    env = FrameStack(env, num_stack)
    agents = {
        agent_id: agent_spec.build_agent()
        for agent_id, agent_spec in agent_specs.items()
    }

    # Test whether env.reset returns stacked duplicated observations
    obs = env.reset()
    assert len(obs) == len(agents)
    for agent_id, agent_obs in obs.items():
        rgb = agent_specs[agent_id].interface.rgb
        agent_obs = np.asarray(agent_obs)
        assert agent_obs.shape == (num_stack, rgb.width, rgb.height, 3)
        for i in range(1, num_stack):
            assert np.allclose(agent_obs[i - 1], agent_obs[i])

    # Test whether env.step removes old and appends new observation
    actions = {
        agent_id: agents[agent_id].act(agent_obs) for agent_id, agent_obs in obs.items()
    }
    obs, _, _, _ = env.step(actions)
    assert len(obs) == len(agents)
    for agent_id, agent_obs in obs.items():
        rgb = agent_specs[agent_id].interface.rgb
        agent_obs = np.asarray(agent_obs)
        assert agent_obs.shape == (num_stack, rgb.width, rgb.height, 3)
        for i in range(1, num_stack - 1):
            assert np.allclose(agent_obs[i - 1], agent_obs[i])
        if num_stack > 1:
            assert not np.allclose(agent_obs[-1], agent_obs[-2])

    env.close()
