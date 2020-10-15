import copy

from ray.rllib.env.multi_agent_env import MultiAgentEnv


class Wrapper(MultiAgentEnv):
    def __init__(self, config):
        base_env_cls = config["base_env_cls"]
        self.env = base_env_cls(config)
        self._agent_keys = list(config["agent_specs"].keys())
        self._last_observations = {k: None for k in self._agent_keys}

    def _get_observations(self, observations):
        return observations

    def _get_rewards(self, last_observations, observations, rewards):
        return rewards

    def _get_infos(self, observations, rewards, infos):
        return infos

    def _update_last_observation(self, observations):
        for agent_id, obs in observations.items():
            self._last_observations[agent_id] = copy.copy(obs)

    def step(self, agent_actions):
        return self.env.step(agent_actions)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def close(self):
        self.env.close()
