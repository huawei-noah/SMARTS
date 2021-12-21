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
# The author of this file is: https://github.com/mg2015started

import numpy as np
import tensorflow as tf
from config import HyperParameters

from utils import OU

tf.compat.v1.disable_eager_execution()


class ActorNetwork:
    def __init__(self, name):
        self.config = HyperParameters()
        self.all_state_size = self.config.all_state_size
        self.action_size = self.config.action_size
        self.tau = self.config.tau

        initial_learning_rate = self.config.lra
        global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.compat.v1.train.exponential_decay(
            initial_learning_rate,
            global_step=global_step,
            decay_steps=200000,
            decay_rate=0.99,
            staircase=True,
        )
        self.optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate)

        self.state_inputs, self.actor_variables, self.action = self.build_actor_network(
            name
        )
        (
            self.state_inputs_target,
            self.actor_variables_target,
            self.action_target,
        ) = self.build_actor_network(name + "_target")

        self.action_gradients = tf.compat.v1.placeholder(
            tf.float32, [None, self.action_size], name="action_gradients"
        )
        self.actor_gradients = tf.gradients(
            self.action, self.actor_variables, -self.action_gradients
        )
        self.optimize = self.optimizer.apply_gradients(
            zip(self.actor_gradients, self.actor_variables)
        )  # global_step=global_step

        self.update_target_op = [
            self.actor_variables_target[i].assign(
                tf.multiply(self.actor_variables[i], self.tau)
                + tf.multiply(self.actor_variables_target[i], 1 - self.tau)
            )
            for i in range(len(self.actor_variables))
        ]

    def split_input(self, all_state):
        # state:[batch, ego_feature_num + npc_feature_num*npc_num + mask]
        env_state = all_state[
            :,
            0 : self.config.ego_feature_num
            + self.config.npc_num * self.config.npc_feature_num,
        ]  # Dims: batch, (ego+npcs)features
        ego_state = tf.reshape(
            env_state[:, 0 : self.config.ego_feature_num],
            [-1, self.config.ego_feature_num],
        )  # Dims: batch, 1, features
        npc_state = tf.reshape(
            env_state[:, self.config.ego_feature_num :],
            [-1, self.config.npc_num, self.config.npc_feature_num],
        )  # Dims: batch, entities, features
        return ego_state, npc_state

    def build_actor_network(self, name):
        with tf.compat.v1.variable_scope(name):
            state_inputs = tf.compat.v1.placeholder(
                tf.float32, [None, self.all_state_size], name="state_inputs"
            )
            ego_state, npc_state = self.split_input(state_inputs)
            # calculate action
            ego_encoder_1 = tf.compat.v1.layers.dense(
                inputs=ego_state,
                units=64,
                activation=tf.nn.tanh,
                kernel_initializer=tf.compat.v1.variance_scaling_initializer(),
                name="ego_encoder_1",
            )
            ego_encoder_2 = tf.compat.v1.layers.dense(
                inputs=ego_encoder_1,
                units=64,
                activation=tf.nn.tanh,
                kernel_initializer=tf.compat.v1.variance_scaling_initializer(),
                name="ego_encoder_2",
            )
            npc_encoder_1 = tf.compat.v1.layers.dense(
                inputs=npc_state,
                units=64,
                activation=tf.nn.tanh,
                kernel_initializer=tf.compat.v1.variance_scaling_initializer(),
                name="npc_encoder_1",
            )
            npc_encoder_2 = tf.compat.v1.layers.dense(
                inputs=npc_encoder_1,
                units=64,
                activation=tf.nn.tanh,
                kernel_initializer=tf.compat.v1.variance_scaling_initializer(),
                name="npc_encoder_2",
            )
            concat_1 = tf.concat(
                [npc_encoder_2[:, i] for i in range(5)], axis=1, name="concat_1"
            )
            concat_2 = tf.compat.v1.layers.dense(
                inputs=concat_1,
                units=256,
                activation=tf.nn.tanh,
                kernel_initializer=tf.compat.v1.variance_scaling_initializer(),
                name="concat_2",
            )
            fc_1 = tf.concat([ego_encoder_2, concat_2], axis=1, name="fc_1")
            fc_2 = tf.compat.v1.layers.dense(
                inputs=fc_1,
                units=256,
                activation=tf.nn.tanh,
                kernel_initializer=tf.compat.v1.variance_scaling_initializer(),
                name="fc_2",
            )
            # action output layer
            action_1 = tf.compat.v1.layers.dense(
                inputs=fc_2,
                units=256,
                activation=tf.nn.tanh,
                kernel_initializer=tf.compat.v1.variance_scaling_initializer(),
                name="action_1",
            )
            action_2 = tf.compat.v1.layers.dense(
                inputs=action_1,
                units=256,
                activation=tf.nn.tanh,
                kernel_initializer=tf.compat.v1.variance_scaling_initializer(),
                name="action_2",
            )
            # action output
            speed_up = tf.compat.v1.layers.dense(
                inputs=action_2,
                units=1,
                activation=tf.nn.sigmoid,
                kernel_initializer=tf.compat.v1.variance_scaling_initializer(),
                name="speed_up",
            )
            slow_down = tf.compat.v1.layers.dense(
                inputs=action_2,
                units=1,
                activation=tf.nn.sigmoid,
                kernel_initializer=tf.compat.v1.variance_scaling_initializer(),
                name="slow_down",
            )
            action = tf.concat([speed_up, slow_down], axis=1, name="action")
        actor_variables = tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=name
        )
        return state_inputs, actor_variables, tf.squeeze(action)

    def get_action(self, sess, state):
        if len(state.shape) < 2:
            state = state.reshape((1, *state.shape))
        action = sess.run(self.action, feed_dict={self.state_inputs: state})
        return action

    def get_action_noise(self, sess, state, rate=1):
        if rate < 0:
            rate = 0
        action = self.get_action(sess, state)
        speed_up_noised = (
            action[0] + OU(action[0], mu=0.6, theta=0.15, sigma=0.3) * rate
        )
        slow_down_noised = (
            action[1] + OU(action[1], mu=0.2, theta=0.15, sigma=0.05) * rate
        )
        action_noise = np.squeeze(
            np.array(
                [
                    np.clip(speed_up_noised, 0.01, 0.99),
                    np.clip(slow_down_noised, 0.01, 0.99),
                ]
            )
        )
        return action_noise

    def get_action_target(self, sess, state):
        action_target = sess.run(
            self.action_target, feed_dict={self.state_inputs_target: state}
        )

        target_noise = 0.01
        action_target_smoothing = (
            action_target + np.random.rand(self.action_size) * target_noise
        )
        speed_up_smoothing = np.clip(action_target_smoothing[:, 0], 0.01, 0.99)
        speed_up_smoothing = speed_up_smoothing.reshape((*speed_up_smoothing.shape, 1))

        slow_down_smoothing = np.clip(action_target_smoothing[:, 1], 0.01, 0.99)
        slow_down_smoothing = slow_down_smoothing.reshape(
            (*slow_down_smoothing.shape, 1)
        )

        action_target_smoothing = np.concatenate(
            [speed_up_smoothing, slow_down_smoothing], axis=1
        )
        return action_target_smoothing

    def train(self, sess, state, action_gradients):
        sess.run(
            self.optimize,
            feed_dict={
                self.state_inputs: state,
                self.action_gradients: action_gradients,
            },
        )

    def update_target(self, sess):
        sess.run(self.update_target_op)


class CriticNetwork:
    def __init__(self, name):
        self.config = HyperParameters()
        self.all_state_size = self.config.all_state_size
        self.action_size = self.config.action_size
        self.tau = self.config.tau

        initial_learning_rate = self.config.lrc
        global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.compat.v1.train.exponential_decay(
            initial_learning_rate,
            global_step=global_step,
            decay_steps=200000,
            decay_rate=0.99,
            staircase=True,
        )
        self.optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate)
        self.optimizer_2 = tf.compat.v1.train.AdamOptimizer(self.learning_rate)

        (
            self.state_inputs,
            self.action,
            self.critic_variables,
            self.q_value,
        ) = self.build_critic_network(name)
        (
            self.state_inputs_target,
            self.action_target,
            self.critic_variables_target,
            self.q_value_target,
        ) = self.build_critic_network(name + "_target")

        self.target = tf.compat.v1.placeholder(tf.float32, [None])
        self.ISWeights = tf.compat.v1.placeholder(tf.float32, [None, 1])
        self.absolute_errors = tf.abs(
            self.target - self.q_value
        )  # for updating sumtree
        self.action_gradients = tf.gradients(self.q_value, self.action)

        self.loss = tf.reduce_mean(
            self.ISWeights
            * tf.compat.v1.losses.huber_loss(
                labels=self.target, predictions=self.q_value
            )
        )
        self.loss_2 = tf.reduce_mean(
            tf.compat.v1.losses.huber_loss(labels=self.target, predictions=self.q_value)
        )
        self.optimize = self.optimizer.minimize(self.loss)  # global_step=global_step
        self.optimize_2 = self.optimizer_2.minimize(self.loss_2)

        self.update_target_op = [
            self.critic_variables_target[i].assign(
                tf.multiply(self.critic_variables[i], self.tau)
                + tf.multiply(self.critic_variables_target[i], 1 - self.tau)
            )
            for i in range(len(self.critic_variables))
        ]

    def split_input(self, all_state):  # state:[batch, 31]
        # state:[batch, ego_feature_num + npc_feature_num*npc_num + mask]
        env_state = all_state[
            :,
            0 : self.config.ego_feature_num
            + self.config.npc_num * self.config.npc_feature_num,
        ]  # Dims: batch, (ego+npcs)features
        ego_state = tf.reshape(
            env_state[:, 0 : self.config.ego_feature_num],
            [-1, self.config.ego_feature_num],
        )  # Dims: batch, 1, features
        npc_state = tf.reshape(
            env_state[:, self.config.ego_feature_num :],
            [-1, self.config.npc_num, self.config.npc_feature_num],
        )  # Dims: batch, entities, features
        return ego_state, npc_state

    def build_critic_network(self, name):
        with tf.compat.v1.variable_scope(name):
            state_inputs = tf.compat.v1.placeholder(
                tf.float32, [None, self.all_state_size], name="state_inputs"
            )
            action_inputs = tf.compat.v1.placeholder(
                tf.float32, [None, self.action_size], name="action_inputs"
            )
            ego_state, npc_state = self.split_input(state_inputs)
            # calculate q-value
            encoder_1 = tf.compat.v1.layers.dense(
                inputs=npc_state,
                units=64,
                activation=tf.nn.tanh,
                kernel_initializer=tf.compat.v1.variance_scaling_initializer(),
                name="encoder_1",
            )
            encoder_2 = tf.compat.v1.layers.dense(
                inputs=encoder_1,
                units=64,
                activation=tf.nn.tanh,
                kernel_initializer=tf.compat.v1.variance_scaling_initializer(),
                name="encoder_2",
            )
            concat = tf.concat(
                [encoder_2[:, i] for i in range(5)], axis=1, name="concat"
            )
            # converge
            fc_1 = tf.concat([ego_state, concat], axis=1, name="fc_1")
            fc_2 = tf.compat.v1.layers.dense(
                inputs=fc_1,
                units=256,
                activation=tf.nn.tanh,
                kernel_initializer=tf.compat.v1.variance_scaling_initializer(),
                name="fc_2",
            )
            # state+action merge
            action_fc = tf.compat.v1.layers.dense(
                inputs=action_inputs,
                units=256,
                activation=tf.nn.tanh,
                kernel_initializer=tf.compat.v1.variance_scaling_initializer(),
                name="action_fc",
            )
            merge = tf.concat([fc_2, action_fc], axis=1, name="merge")
            merge_fc = tf.compat.v1.layers.dense(
                inputs=merge,
                units=256,
                activation=tf.nn.tanh,
                kernel_initializer=tf.compat.v1.variance_scaling_initializer(),
                name="merge_fc",
            )
            # q value output
            q_value = tf.compat.v1.layers.dense(
                inputs=merge_fc,
                units=1,
                activation=None,
                kernel_initializer=tf.compat.v1.variance_scaling_initializer(),
                name="q_value",
            )
        critic_variables = tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=name
        )
        return state_inputs, action_inputs, critic_variables, tf.squeeze(q_value)

    def get_q_value_target(self, sess, state, action_target):
        return sess.run(
            self.q_value_target,
            feed_dict={
                self.state_inputs_target: state,
                self.action_target: action_target,
            },
        )

    def get_gradients(self, sess, state, action):
        return sess.run(
            self.action_gradients,
            feed_dict={self.state_inputs: state, self.action: action},
        )

    def train(self, sess, state, action, target, ISWeights):
        _, _, loss, absolute_errors = sess.run(
            [self.optimize, self.optimize_2, self.loss, self.absolute_errors],
            feed_dict={
                self.state_inputs: state,
                self.action: action,
                self.target: target,
                self.ISWeights: ISWeights,
            },
        )
        return loss, absolute_errors

    def update_target(self, sess):
        sess.run(self.update_target_op)
