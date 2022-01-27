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

from typing import Any, Tuple

import gym


class SingleAgent(gym.Wrapper):
    """Converts a single-agent SMARTS environment's step and reset output to be
    compliant with gym spaces."""

    def __init__(self, env: gym.Env):
        """
        Args:
            env (gym.Env): Single-agent SMARTS environment to be wrapped.
        """
        super(SingleAgent, self).__init__(env)

        agent_ids = list(env.agent_specs.keys())
        assert (
            len(agent_ids) == 1
        ), f"Expected env to have a single agent, but got {len(agent_ids)} agents."
        self._agent_id = agent_ids[0]

        if self.observation_space:
            self.observation_space = self.observation_space[self._agent_id]

    def step(self, action: Any) -> Tuple[Any, float, bool, Any]:
        """Steps a single-agent SMARTS environment.

        Args:
            action (Any): Agent's action

        Returns:
            Tuple[Any, float, bool, Any]: Agent's observation, reward, done, and info
        """
        obs, reward, done, info = self.env.step({self._agent_id: action})
        return (
            obs[self._agent_id],
            reward[self._agent_id],
            done[self._agent_id],
            info[self._agent_id],
        )

    def reset(self) -> Any:
        """Resets a single-agent SMARTS environment.

        Returns:
            Any: Agent's observation
        """
        obs = self.env.reset()
        return obs[self._agent_id]
