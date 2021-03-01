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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# =========================
# Original version: https://github.com/ray-project/ray/blob/master/rllib/contrib/maddpg/maddpg_policy.py
# See ray in THIRD_PARTY_OPEN_SOURCE_SOFTWARE_NOTICE
import logging

import gym
import numpy as np
from ray.rllib.agents.dqn.dqn_tf_policy import _adjust_nstep, minimize_and_clip
from ray.rllib.contrib.maddpg.maddpg import DEFAULT_CONFIG
from ray.rllib.evaluation.metrics import LEARNER_STATS_KEY
from ray.rllib.models import ModelCatalog
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_policy import TFPolicy
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf, try_import_tfp

logger = logging.getLogger(__name__)

tf1, tf, tfv = try_import_tf()
tfp = try_import_tfp()


class MADDPGPostprocessing:
    """Implements agentwise termination signal and n-step learning."""

    @override(Policy)
    def postprocess_trajectory(
        self, sample_batch, other_agent_batches=None, episode=None
    ):
        # FIXME: Get done from info is required since agent wise done is not
        # supported now.
        # sample_batch.data[SampleBatch.DONES] = self.get_done_from_info(sample_batch.data["infos"])

        # N-step Q adjustments
        if self.config["n_step"] > 1:
            _adjust_nstep(
                self.config["n_step"],
                self.config["gamma"],
                sample_batch[SampleBatch.CUR_OBS],
                sample_batch[SampleBatch.ACTIONS],
                sample_batch[SampleBatch.REWARDS],
                sample_batch[SampleBatch.NEXT_OBS],
                sample_batch[SampleBatch.DONES],
            )

        return sample_batch


def _make_continuous_space(space):
    if isinstance(space, gym.spaces.Box):
        return space
    elif isinstance(space, gym.spaces.Discrete):
        return gym.spaces.Box(low=np.zeros((space.n,)), high=np.ones((space.n,)))
    else:
        return get_preprocessor(space)(space).observation_space


def _make_ph_n(space_n, name=""):
    return [
        tf1.placeholder(tf.float32, shape=(None,) + space.shape, name=name + "_%d" % i)
        for i, space in enumerate(space_n)
    ]


def _make_set_weight_op(variables):
    vs = list()
    for v in variables.values():
        vs += v
    phs = [
        tf1.placeholder(
            tf.float32, shape=v.get_shape(), name=v.name.split(":")[0] + "_ph"
        )
        for v in vs
    ]
    return tf.group(*[v.assign(ph) for v, ph in zip(vs, phs)]), phs


def _make_loss_inputs(placeholders):
    return [(ph.name.split("/")[-1].split(":")[0], ph) for ph in placeholders]


def _make_target_update_op(vs, target_vs, tau):
    return [
        target_v.assign(tau * v + (1.0 - tau) * target_v)
        for v, target_v in zip(vs, target_vs)
    ]


class MADDPG2TFPolicy(MADDPGPostprocessing, TFPolicy):
    def __init__(self, obs_space, act_space, config):
        # _____ Initial Configuration
        config = dict(DEFAULT_CONFIG, **config)
        self.config = config
        self.global_step = tf1.train.get_or_create_global_step()

        # FIXME: Get done from info is required since agentwise done is not
        # supported now.
        # self.get_done_from_info = np.vectorize(lambda info: info.get("done", False))

        agent_id = config["agent_id"]
        if agent_id is None:
            raise ValueError("Must set `agent_id` in the policy config.")

        obs_space_n, act_space_n = [], []
        self.agent_ids = []

        for pid, (_, obs_space, act_space, _) in config["multiagent"][
            "policies"
        ].items():
            # assert isinstance(obs_space, gym.spaces.Box), obs_space
            obs_space_n.append(_make_continuous_space(obs_space))
            act_space_n.append(_make_continuous_space(act_space))
            self.agent_ids.append(pid)

        self.agent_idx = self.agent_ids.index(agent_id)

        # _____ Placeholders
        # Placeholders for policy evaluation and updates
        obs_ph_n = _make_ph_n(obs_space_n, "obs")
        act_ph_n = _make_ph_n(act_space_n, "actions")
        new_obs_ph_n = _make_ph_n(obs_space_n, "new_obs")
        new_act_ph_n = _make_ph_n(act_space_n, "new_actions")
        rew_ph = tf1.placeholder(
            tf.float32, shape=None, name="rewards_{}".format(self.agent_idx)
        )
        done_ph = tf1.placeholder(
            tf.float32, shape=None, name="dones_{}".format(self.agent_idx)
        )

        if config["use_local_critic"]:  # no global ...
            obs_space_n, act_space_n = (
                [obs_space_n[self.agent_idx]],
                [act_space_n[self.agent_idx]],
            )
            obs_ph_n, act_ph_n = [obs_ph_n[self.agent_idx]], [act_ph_n[self.agent_idx]]
            new_obs_ph_n, new_act_ph_n = (
                [new_obs_ph_n[self.agent_idx]],
                [new_act_ph_n[self.agent_idx]],
            )
            self.agent_idx = 0

        # _____ Value Network
        # Build critic network for t.
        critic, _, critic_model_n, critic_vars = self._build_critic_network(
            obs_ph_n,
            act_ph_n,
            obs_space_n,
            act_space_n,
            config["use_state_preprocessor"],
            config["critic_hiddens"],
            getattr(tf.nn, config["critic_hidden_activation"]),
            scope="critic",
        )

        # Build critic network for t + 1.
        target_critic, _, _, target_critic_vars = self._build_critic_network(
            new_obs_ph_n,
            new_act_ph_n,
            obs_space_n,
            act_space_n,
            config["use_state_preprocessor"],
            config["critic_hiddens"],
            getattr(tf.nn, config["critic_hidden_activation"]),
            scope="target_critic",
        )

        # Build critic loss.
        td_error = tf.subtract(
            tf.stop_gradient(
                rew_ph
                + (1.0 - done_ph)
                * (config["gamma"] ** config["n_step"])
                * target_critic[:, 0]
            ),
            critic[:, 0],
        )
        critic_loss = tf.reduce_mean(input_tensor=td_error ** 2)

        # _____ Policy Network
        # Build actor network for t.
        act_sampler, actor_feature, actor_model, actor_vars = self._build_actor_network(
            obs_ph_n[self.agent_idx],
            obs_space_n[self.agent_idx],
            act_space_n[self.agent_idx],
            config["use_state_preprocessor"],
            config["actor_hiddens"],
            getattr(tf.nn, config["actor_hidden_activation"]),
            scope="actor",
        )

        # Build actor network for t + 1.
        self.new_obs_ph = new_obs_ph_n[self.agent_idx]
        self.target_act_sampler, _, _, target_actor_vars = self._build_actor_network(
            self.new_obs_ph,
            obs_space_n[self.agent_idx],
            act_space_n[self.agent_idx],
            config["use_state_preprocessor"],
            config["actor_hiddens"],
            getattr(tf.nn, config["actor_hidden_activation"]),
            scope="target_actor",
        )

        # Build actor loss.
        act_n = act_ph_n.copy()
        act_n[self.agent_idx] = act_sampler
        critic, _, _, _ = self._build_critic_network(
            obs_ph_n,
            act_n,
            obs_space_n,
            act_space_n,
            config["use_state_preprocessor"],
            config["critic_hiddens"],
            getattr(tf.nn, config["critic_hidden_activation"]),
            scope="critic",
        )
        actor_loss = -tf.reduce_mean(input_tensor=critic)
        if config["actor_feature_reg"] is not None:
            actor_loss += config["actor_feature_reg"] * tf.reduce_mean(
                input_tensor=actor_feature ** 2
            )

        # _____ Losses
        self.losses = {"critic": critic_loss, "actor": actor_loss}

        # _____ Optimizers
        self.optimizers = {
            "critic": tf1.train.AdamOptimizer(config["critic_lr"]),
            "actor": tf1.train.AdamOptimizer(config["actor_lr"]),
        }

        # _____ Build variable update ops.
        self.tau = tf1.placeholder_with_default(config["tau"], shape=(), name="tau")
        self.update_target_vars = _make_target_update_op(
            critic_vars + actor_vars, target_critic_vars + target_actor_vars, self.tau
        )

        self.vars = {
            "critic": critic_vars,
            "actor": actor_vars,
            "target_critic": target_critic_vars,
            "target_actor": target_actor_vars,
        }
        self.update_vars, self.vars_ph = _make_set_weight_op(self.vars)

        # _____ TensorFlow Initialization

        self.sess = tf1.get_default_session()
        loss_inputs = _make_loss_inputs(
            obs_ph_n + act_ph_n + new_obs_ph_n + new_act_ph_n + [rew_ph, done_ph]
        )

        TFPolicy.__init__(
            self,
            obs_space,
            act_space,
            config=config,
            sess=self.sess,
            obs_input=obs_ph_n[self.agent_idx],
            sampled_action=act_sampler,
            loss=actor_loss + critic_loss,
            loss_inputs=loss_inputs,
            dist_inputs=actor_feature,
        )

        self.sess.run(tf1.global_variables_initializer())

        # Hard initial update
        self.update_target(1.0)

    @override(TFPolicy)
    def optimizer(self):
        return None

    @override(TFPolicy)
    def gradients(self, optimizer, loss):
        if self.config["grad_norm_clipping"] is not None:
            self.gvs = {
                k: minimize_and_clip(
                    optimizer,
                    self.losses[k],
                    self.vars[k],
                    self.config["grad_norm_clipping"],
                )
                for k, optimizer in self.optimizers.items()
            }
        else:
            self.gvs = {
                k: optimizer.compute_gradients(self.losses[k], self.vars[k])
                for k, optimizer in self.optimizers.items()
            }
        return self.gvs["critic"] + self.gvs["actor"]

    @override(TFPolicy)
    def build_apply_op(self, optimizer, grads_and_vars):
        critic_apply_op = self.optimizers["critic"].apply_gradients(self.gvs["critic"])

        with tf1.control_dependencies([tf1.assign_add(self.global_step, 1)]):
            with tf1.control_dependencies([critic_apply_op]):
                actor_apply_op = self.optimizers["actor"].apply_gradients(
                    self.gvs["actor"]
                )

        return actor_apply_op

    @override(TFPolicy)
    def extra_compute_action_feed_dict(self):
        return {}

    @override(TFPolicy)
    def extra_compute_grad_fetches(self):
        return {LEARNER_STATS_KEY: {}}

    @override(TFPolicy)
    def get_weights(self):
        var_list = []
        for var in self.vars.values():
            var_list += var
        return self.sess.run(var_list)

    @override(TFPolicy)
    def set_weights(self, weights):
        self.sess.run(self.update_vars, feed_dict=dict(zip(self.vars_ph, weights)))

    @override(Policy)
    def get_state(self):
        return TFPolicy.get_state(self)

    @override(Policy)
    def set_state(self, state):
        TFPolicy.set_state(self, state)

    def _build_critic_network(
        self,
        obs_n,
        act_n,
        obs_space_n,
        act_space_n,
        use_state_preprocessor,
        hiddens,
        activation=None,
        scope=None,
    ):
        """Build critic network

        Args:
            obs_n: list, the observation placeholder list contains at least one.
            act_n: list, the action placeholder list contains at least one.
            obs_space_n: list, the observation space list contains at least one.
            act_space_n: list, the action space list contains at least one.
            use_state_preprocessor: bool, if true, there are `n` preprocessor models for each observation placeholder
                otherwise, no.
            hiddens: list, a list of unit definition.
            activation: tf.nn, default is None, to initialize the activation function.
            scope: str, name the variable scope

        Returns:
            out: tf.Tensor, logits out.
            feature: tf.Tensor, intputs of logits output.
            model_n: list, preprocessor models for observation inputs.
            variables: list, return global variables of this critic network.
        """

        with tf1.variable_scope(scope, reuse=tf1.AUTO_REUSE) as scope:
            if use_state_preprocessor:
                model_n = [
                    ModelCatalog.get_model(
                        {
                            "obs": obs,
                            "is_training": self._get_is_training_placeholder(),
                        },
                        obs_space,
                        act_space,
                        1,
                        self.config["model"],
                    )
                    for obs, obs_space, act_space in zip(
                        obs_n, obs_space_n, act_space_n
                    )
                ]
                out_n = [model.last_layer for model in model_n]
                out = tf.concat(out_n + act_n, axis=1)
            else:
                model_n = [None] * len(obs_n)
                out = tf.concat(obs_n + act_n, axis=1)

            for hidden in hiddens:
                out = tf1.layers.dense(out, units=hidden, activation=activation)
            feature = out
            out = tf1.layers.dense(feature, units=1, activation=None)

        return out, feature, model_n, tf1.global_variables(scope.name)

    def _build_actor_network(
        self,
        obs,
        obs_space,
        act_space,
        use_state_preprocessor,
        hiddens,
        activation=None,
        scope=None,
    ):
        with tf1.variable_scope(scope, reuse=tf1.AUTO_REUSE) as scope:
            if use_state_preprocessor:
                model = ModelCatalog.get_model(
                    {
                        "obs": obs,
                        "is_training": self._get_is_training_placeholder(),
                    },
                    obs_space,
                    act_space,
                    1,
                    self.config["model"],
                )
                out = model.last_layer
            else:
                model = None
                out = obs

            for hidden in hiddens:
                out = tf1.layers.dense(out, units=hidden, activation=activation)
            feature = tf1.layers.dense(out, units=act_space.shape[0], activation=None)
            sampler = tfp.distributions.RelaxedOneHotCategorical(
                temperature=1.0, logits=feature
            ).sample()

        return sampler, feature, model, tf1.global_variables(scope.name)

    def update_target(self, tau=None):
        if tau is not None:
            self.sess.run(self.update_target_vars, {self.tau: tau})
        else:
            self.sess.run(self.update_target_vars)
