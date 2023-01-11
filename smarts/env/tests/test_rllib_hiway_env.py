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
from pathlib import Path

import gym
import numpy as np
import pytest

from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.env.rllib_hiway_env import RLlibHiWayEnv
from smarts.zoo.agent_spec import AgentSpec

AGENT_ID = "Agent-007"


@pytest.fixture
def rllib_agent():
    return {
        "agent_spec": AgentSpec(
            interface=AgentInterface.from_type(
                AgentType.Standard,
                # We use a low number of steps here since this is a test
                max_episode_steps=10,
            ),
        ),
        "action_space": gym.spaces.Box(
            low=np.array([0.0, 0.0, -1.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32,
        ),
    }


def test_rllib_hiway_env(rllib_agent):
    # XXX: We should be able to simply provide "scenarios/sumo/loop"?
    scenario_path = Path(__file__).parent / "../../../scenarios/sumo/loop"

    env_config = {
        "scenarios": [str(scenario_path.absolute())],
        "seed": 42,
        "headless": True,
        "agent_specs": {AGENT_ID: rllib_agent["agent_spec"]},
    }

    class atdict(dict):
        __getattr__ = dict.__getitem__
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__

    env = RLlibHiWayEnv(config=atdict(**env_config, worker_index=0, vector_index=1))
    agent_ids = list(env_config["agent_specs"].keys())

    dones = {"__all__": False}
    env.reset()
    while not dones["__all__"]:
        _, _, dones, _ = env.step(
            {aid: rllib_agent["action_space"].sample() for aid in agent_ids}
        )
    env.close()
