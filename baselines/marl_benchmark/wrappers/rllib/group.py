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
import copy

from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.env.constants import GROUP_INFO, GROUP_REWARDS
from ray.rllib.env.group_agents_wrapper import _GroupAgentsWrapper


class Group(_GroupAgentsWrapper):
    def __init__(self, config):
        obs_space = config["obs_space"]
        act_space = config["act_space"]
        groups = config["groups"]

        base_env_cls = config["base_env_cls"]
        env = base_env_cls(config)

        self.observation_adapter = config.get("observation_adapter")
        self.info_adapter = config.get("info_adapter")
        self.reward_adapter = config.get("reward_adapter")

        self._agent_keys = list(config["agent_specs"].keys())
        self._last_observations = {k: None for k in self._agent_keys}

        super(Group, self).__init__(env, groups, obs_space, act_space)

    def _get_infos(self, obs, rewards, infos):
        return infos

    def _get_rewards(self, last_obs, obs, rewards):
        res = {}
        for key in obs:
            res[key] = self.reward_adapter(last_obs[key], obs[key], rewards[key])
        return res

    def _get_observations(self, obs):
        res = {}
        for key, _obs in obs.items():
            res[key] = self.observation_adapter(_obs)
        return res

    def _update_last_observation(self, obs):
        for k, _obs in obs.items():
            self._last_observations[k] = copy.copy(_obs)

    def step(self, action_dict):
        action_dict = self._ungroup_items(action_dict)
        obs, rewards, dones, infos = self.env.step(action_dict)

        infos = self._get_infos(obs, rewards, infos)
        rewards = self._get_rewards(self._last_observations, obs, rewards)
        self._update_last_observation(obs)
        obs = self._get_observations(obs)

        # Apply grouping transforms to the env outputs
        obs = self._group_items(obs)
        rewards = self._group_items(rewards, agg_fn=lambda gvals: list(gvals.values()))
        dones = self._group_items(dones, agg_fn=lambda gvals: any(gvals.values()))
        infos = self._group_items(
            infos, agg_fn=lambda gvals: {GROUP_INFO: list(gvals.values())}
        )

        # Aggregate rewards, but preserve the original values in infos
        for agent_id, rew in rewards.items():
            if isinstance(rew, list):
                rewards[agent_id] = sum(rew)
                if agent_id not in infos:
                    infos[agent_id] = {}
                infos[agent_id][GROUP_REWARDS] = rew

        dones["__all__"] = any(dones.values())
        return obs, rewards, dones, infos

    def reset(self):
        obs = self.env.reset()
        self._update_last_observation(obs)
        obs = self._get_observations(obs)
        return self._group_items(obs)
