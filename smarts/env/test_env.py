# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
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
from typing import Any, Dict, Optional, Sequence, Tuple, Union
import gymnasium as gym
import random

from smarts.zoo.agent_spec import AgentSpec
from gymnasium.spaces import Discrete
from gymnasium.spaces import Tuple as gym_Tuple
from gymnasium.spaces import Dict as gym_Dict


class test_env(gym.Env):
    """test environment for gymnasium"""
    def __init__(
        self,
        agent_interfaces: Dict[str, AgentSpec],
    ):
        agent_ids = list(agent_interfaces.keys())
        assert (
            len(agent_ids) == 1
        ), f"Expected env to have a single agent, but got {len(agent_ids)} agents."
        self.agent_id = agent_ids[0]
        self.action_space = gym_Dict(
            {self.agent_id: gym_Tuple((Discrete(61), Discrete(4, start=-1)))}
        )
        self.observation_space = gym_Dict({self.agent_id: Discrete(4, start=10)})
        self.agent_specs = agent_interfaces

    def step(
        self, agent_actions: Dict
    ) -> Dict:
        """Environment step"""
        if list(agent_actions.values())[0][0] < 30:
            return (
                {self.agent_id: 10},
                {self.agent_id: 1},
                {self.agent_id: False},
                {self.agent_id: False},
                {self.agent_id: {}},
            )
        else:
            return (
                {self.agent_id: 13},
                {self.agent_id: 2},
                {self.agent_id: True},
                {self.agent_id: True},
                {self.agent_id: {}},
            )

    def reset(self, seed=None, options={}) -> Dict:
        """Environment reset"""
        super().reset(seed=seed)
        return {self.agent_id: ({self.agent_id: 10}, {self.agent_id: {}})}

    def render(self):
        """Environment render"""
        return None
