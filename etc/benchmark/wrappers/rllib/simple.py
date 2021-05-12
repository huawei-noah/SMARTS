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
from benchmark.wrappers.rllib.wrapper import Wrapper


class Simple(Wrapper):
    def __init__(self, config):
        super(Simple, self).__init__(config)

        self.observation_adapter = config["observation_adapter"]
        self.info_adapter = config.get("info_adapter")
        self.reward_adapter = config["reward_adapter"]

    def _get_rewards(self, last_observation, observation, reward):
        res = {}
        for k in observation:
            res[k] = self.reward_adapter(last_observation[k], observation[k], reward[k])
        return res

    def _get_observations(self, observations):
        res = {}
        for k, _obs in observations.items():
            res[k] = self.observation_adapter(_obs)
        return res

    def step(self, agent_actions):
        observations, rewards, dones, infos = self.env.step(agent_actions)
        infos = self._get_infos(observations, rewards, infos)
        rewards = self._get_rewards(self._last_observations, observations, rewards)
        self._update_last_observation(observations)  # it is environment observation
        observations = self._get_observations(observations)
        return observations, rewards, dones, infos

    def reset(self):
        obs = self.env.reset()
        self._update_last_observation(obs)
        return self._get_observations(obs)
