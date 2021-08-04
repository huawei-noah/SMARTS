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

from ray.rllib.env import MultiAgentEnv

from smarts.env.rllib_hiway_env import RLlibHiWayEnv


class Wrapper(MultiAgentEnv):
    def __init__(self, config):
        self.env = RLlibHiWayEnv(config)
        self._agent_keys = list(config["agent_specs"].keys())
        self._last_observations = {k: None for k in self._agent_keys}

    @staticmethod
    def get_observation_space(observation_space, wrapper_config):
        raise NotImplementedError

    @staticmethod
    def get_action_space(action_space, wrapper_config=None):
        raise NotImplementedError

    @staticmethod
    def get_preprocessor():
        raise NotImplementedError

    @staticmethod
    def get_observation_adapter(
        observation_space, feature_configs, wrapper_config=None
    ):
        raise NotImplementedError

    @staticmethod
    def get_action_adapter(action_type, action_space, wrapper_config=None):
        raise NotImplementedError

    @staticmethod
    def get_reward_adapter(observation_adapter):
        raise NotImplementedError

    def _get_observations(self, observations):
        return observations

    def _get_rewards(self, observations, rewards):
        return rewards

    def _get_infos(self, env_observations, rewards, infos):
        return infos

    def _update_last_observation(self, observations):
        for agent_id, obs in observations.items():
            self._last_observations[agent_id] = copy.copy(obs)

    def step(self, agent_actions):
        return self.env.step(agent_actions)

    def reset(self):
        return self.env.reset()

    def close(self):
        self.env.close()
