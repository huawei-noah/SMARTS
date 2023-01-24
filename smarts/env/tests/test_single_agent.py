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

from typing import Dict

import gym
import numpy as np
import pytest

from smarts.core.agent import Agent
from smarts.core.agent_interface import RGB, AgentInterface
from smarts.core.controllers import ActionSpaceType
from smarts.env.wrappers.single_agent import SingleAgent
from smarts.zoo.agent_spec import AgentSpec


def _make_agent_specs(num_agent):
    agent_specs: Dict[str, AgentSpec] = {
        "AGENT_"
        + str(agent_id): AgentSpec(
            interface=AgentInterface(
                top_down_rgb=RGB(),
                action=ActionSpaceType.Lane,
            ),
            agent_builder=lambda: Agent.from_function(lambda _: "keep_lane"),
        )
        for agent_id in range(num_agent)
    }

    return agent_specs


class ObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_space = {}
        for agent_id, agent_spec in self.env.agent_specs.items():
            rgb: RGB = (
                agent_spec.interface.rgb
            )  # pytype: disable=annotation-type-mismatch
            obs_space[agent_id] = gym.spaces.Box(
                low=0,
                high=255,
                shape=(
                    rgb.width,
                    rgb.height,
                    3,
                ),
                dtype=np.uint8,
            )
        self.observation_space = gym.spaces.Dict(obs_space)

    def observation(self, obs):
        return {
            agent_name: agent_obs.top_down_rgb.data
            for agent_name, agent_obs in obs.items()
        }


@pytest.fixture
def base_env(request):
    agent_specs = _make_agent_specs(request.param)
    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=["scenarios/sumo/figure_eight"],
        agent_specs=agent_specs,
        headless=True,
        visdom=False,
        fixed_timestep_sec=0.01,
    )
    env = ObservationWrapper(env)

    yield env
    env.close()


@pytest.mark.parametrize("base_env", [1, 2], indirect=True)
def test_init(base_env):
    # Compute multiagent specs and space, i.e., ma_*
    ma_agent_specs = base_env.agent_specs
    ma_obs_space = base_env.observation_space

    # Test wrapping an env containing one and more than one agent
    if len(ma_agent_specs) > 1:
        with pytest.raises(AssertionError):
            env = SingleAgent(base_env)
            env.close()
        return
    else:
        env = SingleAgent(base_env)

    # Test env observation space
    agent_id = next(iter(ma_agent_specs.keys()))
    assert env.observation_space == ma_obs_space[agent_id]

    env.close()


@pytest.mark.parametrize("base_env", [1], indirect=True)
def test_reset_and_step(base_env):
    ma_agent_specs = base_env.agent_specs
    ma_obs_space = base_env.observation_space
    env = SingleAgent(base_env)

    # Test resetting the env
    obs = env.reset()
    assert isinstance(obs, np.ndarray)
    agent_id = next(iter(ma_agent_specs.keys()))
    assert obs.shape == ma_obs_space[agent_id].shape

    # Test stepping the env
    obs, reward, done, info = env.step("keep_lane")
    assert isinstance(obs, np.ndarray)
    assert obs.shape == ma_obs_space[agent_id].shape
    assert isinstance(reward, float)
    assert type(done) is bool
    assert set(info.keys()) == set(["score", "env_obs", "done"])

    env.close()
