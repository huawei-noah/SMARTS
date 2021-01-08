import copy

from ray.rllib.env.multi_agent_env import MultiAgentEnv
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

    def _get_infos(self, observations, rewards, infos):
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
