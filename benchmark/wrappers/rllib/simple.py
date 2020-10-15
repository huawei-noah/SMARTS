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
