"""
This file contains an RLlib-trained policy evaluation usage (not for training).
"""
import pickle
from pathlib import Path

import gym
import tensorflow as tf
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy as LoadPolicy
from ray.rllib.models import ModelCatalog

from smarts.core.agent import Agent


class RLAgent(Agent):
    def __init__(self, load_path, policy_name, observation_space, action_space):
        self._checkpoint_path = load_path
        self._policy_name = policy_name
        self._observation_space = observation_space
        self._action_space = action_space
        self._sess = None

        if isinstance(action_space, gym.spaces.Box):
            self.is_continuous = True
        elif isinstance(action_space, gym.spaces.Discrete):
            self.is_continuous = False
        else:
            raise TypeError("Unsupport action space")

        if self._sess:
            return

        self._prep = ModelCatalog.get_preprocessor_for_space(self._observation_space)
        self._sess = tf.compat.v1.Session(graph=tf.Graph())
        self._sess.__enter__()

        with tf.name_scope(self._policy_name):
            # obs_space need to be flattened before passed to PPOTFPolicy
            flat_obs_space = self._prep.observation_space
            self.policy = LoadPolicy(flat_obs_space, self._action_space, {})
            objs = pickle.load(open(self._checkpoint_path, "rb"))
            objs = pickle.loads(objs["worker"])
            state = objs["state"]
            weights = state[self._policy_name]
            self.policy.set_weights(weights)

    def act(self, obs):
        if isinstance(obs, list):
            # batch infer
            obs = [self._prep.transform(o) for o in obs]
            action = self.policy.compute_actions(obs, explore=False)[0]
        else:
            # single infer
            obs = self._prep.transform(obs)
            action = self.policy.compute_actions([obs], explore=False)[0][0]

        return action
