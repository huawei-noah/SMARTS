from collections import deque

import numpy as np

from .wrapper import Wrapper


class FrameStack(Wrapper):
    """ By default, this wrapper will stack 4 consecutive frames as an agent obsrvation"""

    def __init__(self, config):
        super(FrameStack, self).__init__(config)

        self.num_stack = config["num_stack"]
        self.observation_adapter = config.get("observation_adapter", None)
        self.info_adapter = config.get("info_adapter", None)
        self.reward_adapter = config.get("reward_adapter", None)

        self.frames = dict.fromkeys(self._agent_keys, deque(maxlen=self.num_stack))

    @staticmethod
    def stack_frames(frames):
        proto = frames[0]

        if isinstance(proto, dict):
            res = dict()
            for key in proto.keys():
                res[key] = np.stack([frame[key] for frame in frames], axis=0)
        elif isinstance(proto, np.ndarray):
            res = np.stack(frames, axis=0)
        else:
            raise NotImplementedError

        return res

    def _get_observations(self, frames):
        """ Update frame stack with given single frames, 
        then return nested array with given agent ids 
        """

        for k, frame in frames.items():
            self.frames[k].append(frame)

        agent_ids = list(frames.keys())
        observations = dict.fromkeys(agent_ids)

        for k in agent_ids:
            observation = list(self.frames[k])
            if self.observation_adapter:
                observation = self.observation_adapter(observation)
            else:
                observation = np.asarray(observation)
            observations[k] = observation

        return observations

    def _get_rewards(self, last_obsrvation, observations, env_rewards):
        if self.reward_adapter:
            agent_ids = list(env_rewards.keys())
            rewards = dict.fromkeys(agent_ids, None)

            for k in agent_ids:
                rewards[k] = self.reward_adapter(
                    list(last_observation[k]), list(self.frames[k]), env_rewards[k]
                )
            return rewards
        else:
            return env_rewards

    def step(self, agent_actions):
        observations, rewards, dones, infos = self.env.step(agent_actions)

        infos = self._get_infos(observations, rewards, infos)
        rewards = self._get_rewards(self._last_observations, observations, rewards)
        observations = self._get_observations(observations)
        self._update_last_observation(self.frames)

        return observations, rewards, dones, infos

    def reset(self):
        observations = self.env.reset()
        for k, observation in observations.items():
            _ = [self.frames[k].append(observation) for _ in range(self.num_stack)]
        self._update_last_observation(self.frames)
        return self._get_observations(observations)
