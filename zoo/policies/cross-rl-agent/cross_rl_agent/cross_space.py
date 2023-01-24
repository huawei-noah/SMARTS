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
# FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# The author of this file is: https://github.com/mg2015started

import heapq

import numpy as np
import tensorflow as tf

from smarts.core.agent_interface import AgentInterface
from smarts.core.controllers import ActionSpaceType

tf.compat.v1.disable_eager_execution()

""" a series of params"""


class HyperParameters(object):
    def __init__(self):
        # Env parameters
        self.ego_feature_num = 4
        self.npc_num = 5
        self.npc_feature_num = 5

        self.state_size = self.ego_feature_num + self.npc_num * self.npc_feature_num
        self.mask_size = self.npc_num + 1
        self.task_size = 4

        self.all_state_size = self.state_size + self.mask_size + self.task_size
        self.action_size = 2

        # Training parameters
        self.noised_episodes = 2500  # 2500
        self.max_steps = 500  # 400
        self.batch_size = 256  # 256
        self.train_frequency = 2

        # Soft update
        self.tau = 1e-3

        # LEARNING hyperparameters
        self.lra = 2e-5
        self.lrc = 1e-4
        self.gamma = 0.99  # Discounting rate


""" Adapters"""


def observation_adapter(env_obs):
    ego_feature_num = 4
    npc_feature_num = 5
    near_npc_number = 5
    mask_size = near_npc_number + 1
    env_state_size = ego_feature_num + near_npc_number * npc_feature_num
    # get ego state
    ego_states = env_obs.ego_vehicle_state
    ego_x = ego_states.position[0]
    ego_y = ego_states.position[1]
    ego_loc = ego_states.position[0:2]
    ego_mission = ego_states.mission
    ego_yaw = ego_states.heading
    ego_speed = ego_states.speed
    # update neighbor vehicle list
    detect_range = 37.5
    veh_within_detect_range_list = []
    for index, vehicle_state in enumerate(env_obs.neighborhood_vehicle_states):
        npc_loc = vehicle_state.position[0:2]
        distance = np.linalg.norm(npc_loc - ego_loc)
        if distance < detect_range:
            add_dict = {"vehicle_state": vehicle_state, "distance": distance}
            veh_within_detect_range_list.append(add_dict)

    r_veh_list = []
    ir_veh_list = []
    # Get relavent npc vehicle
    for veh_dic in veh_within_detect_range_list:
        npc_x = veh_dic["vehicle_state"].position[0]
        npc_y = veh_dic["vehicle_state"].position[1]
        npc_yaw = veh_dic["vehicle_state"].heading

        distance = veh_dic["distance"]
        y_relative = (npc_y - ego_y) * np.cos(ego_yaw) - (npc_x - ego_x) * np.sin(
            ego_yaw
        )

        yaw_relative = npc_yaw - ego_yaw

        if y_relative < -5 or (yaw_relative < 0.1 and distance > 10):
            ir_veh_list.append(veh_dic)
        else:
            r_veh_list.append(veh_dic)

    # sort the vehicles according to their distance
    _near_npc = heapq.nsmallest(
        near_npc_number, r_veh_list, key=lambda s: s["distance"]
    )
    distance_list = []
    for i in range(len(_near_npc)):
        distance_list.append(_near_npc[i]["distance"])
    # print('nearest veh:', distance_list)
    r_npc_list = [x["vehicle_state"] for x in _near_npc]
    ir_npc_list = [x["vehicle_state"] for x in ir_veh_list]

    # get environment state
    env_state = []
    if ego_states.edge_id == "edge-south-SN":  # start lane
        ego_pos_flag = [1, 0, 0]
    elif "junction" in ego_states.edge_id:  # junction
        ego_pos_flag = [0, 1, 0]
    else:  # goal lane
        ego_pos_flag = [0, 0, 1]

    ego_state = ego_pos_flag + [ego_speed]
    # print(ego_states.speed)
    env_state += ego_state
    # print('step')
    for veh_state in r_npc_list:
        # coordinates relative to ego
        npc_x = veh_state.position[0]
        npc_y = veh_state.position[1]
        npc_yaw = veh_state.heading
        x_relative = (npc_y - ego_y) * np.sin(ego_yaw) + (npc_x - ego_x) * np.cos(
            ego_yaw
        )
        y_relative = (npc_y - ego_y) * np.cos(ego_yaw) - (npc_x - ego_x) * np.sin(
            ego_yaw
        )
        # yaw relative to ego
        delta_yaw = npc_yaw - ego_yaw
        # speed
        npc_speed = veh_state.speed
        # state representation for RL
        # print(np.linalg.norm(np.array([x_relative, y_relative])))
        npc_state = [
            x_relative,
            y_relative,
            npc_speed,
            np.cos(delta_yaw),
            np.sin(delta_yaw),
        ]
        # print(ego_x, npc_x, x_relative, ego_y, npc_y, y_relative)

        # intergrate states
        env_state += npc_state

    # get aux state, whichs include task vector & vehicle mask
    mask = list(np.ones(mask_size))
    if len(env_state) < env_state_size:
        zero_padding_num = int((env_state_size - len(env_state)) / npc_feature_num)
        for _ in range(zero_padding_num):
            mask.pop()
        for _ in range(zero_padding_num):
            mask.append(0)
        while len(env_state) < env_state_size:
            env_state.append(0)

    goal_x = ego_mission.goal.position[0]
    if goal_x == 127.6:
        task = [1, 0, 0, 1]
    elif goal_x == 151.6:
        task = [0, 1, 0, 1]
    elif goal_x == 172.4:
        task = [0, 0, 1, 1]
    aux_state = mask + task

    # final state_mask
    total_state = np.array(env_state + aux_state, dtype=np.float32)
    # print(state_mask, state_mask.shape)
    return total_state


def action_adapter(action):
    target_speed = np.clip(action[0] - action[1] / 4, 0, 1)
    target_speed = target_speed * 12

    agent_action = [target_speed, int(0)]
    return agent_action


def reward_adapter(env_obs, reward):
    # set task vector
    goal_x = env_obs.ego_vehicle_state.mission.goal.position[0]
    if goal_x == 127.6:
        task = [1, 0, 0, 1]
    elif goal_x == 151.6:
        task = [0, 1, 0, 1]
    elif goal_x == 172.4:
        task = [0, 0, 1, 1]

    # compute reward vector
    reward_c = 0.0
    reward_s = 0.0
    ego_events = env_obs.events

    # checking
    collision = len(ego_events.collisions) > 0  # check collision
    time_exceed = ego_events.reached_max_episode_steps  # check time exceeds
    reach_goal = ego_events.reached_goal
    # penalty
    reward_c += -0.3  # step cost
    if collision:
        print("collision:", ego_events.collisions)
        print("nearest veh:", observation_adapter(env_obs)[4:9])
        print("Failure. Ego vehicle collides with npc vehicle.")
        reward_s += -650
    elif time_exceed:
        print("nearest veh:", observation_adapter(env_obs)[4:9])
        print("Failure. Time exceed.")
        reward_c += -50
    # reward
    else:
        if reach_goal:
            print("nearest veh:", observation_adapter(env_obs)[4:9])
            print("Success. Ego vehicle reached goal.")
            reward_s += 30

    # reward vector for multi task
    reward = [i * reward_s for i in task[0:3]] + [reward_c]

    return reward


def get_aux_info(env_obs):
    ego_events = env_obs.events
    collision = len(ego_events.collisions) > 0  # check collision
    time_exceed = ego_events.reached_max_episode_steps  # check time exceeds
    reach_goal = ego_events.reached_goal
    if collision:
        aux_info = "collision"
    elif time_exceed:
        aux_info = "time_exceed"
    elif reach_goal:
        aux_info = "success"
    else:
        aux_info = "running"
    return aux_info


cross_interface = AgentInterface(
    max_episode_steps=500,
    neighborhood_vehicle_states=True,
    waypoint_paths=True,
    action=ActionSpaceType.LaneWithContinuousSpeed,
)

""" Network structure"""


class SocMtActorNetwork:
    def __init__(self, name):
        # learning params
        self.config = HyperParameters()
        self.all_state_size = self.config.all_state_size
        self.action_size = self.config.action_size
        self.tau = self.config.tau

        # network params
        self.feature_head = 1
        self.features_per_head = 64
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

        (
            self.state_inputs,
            self.actor_variables,
            self.action,
            self.attention_matrix,
        ) = self.build_actor_network(name)
        (
            self.state_inputs_target,
            self.actor_variables_target,
            self.action_target,
            self.attention_matrix_target,
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
            [-1, 1, self.config.ego_feature_num],
        )  # Dims: batch, 1, features
        npc_state = tf.reshape(
            env_state[:, self.config.ego_feature_num :],
            [-1, self.config.npc_num, self.config.npc_feature_num],
        )  # Dims: batch, entities, features

        aux_state = all_state[:, -(self.config.mask_size + self.config.task_size) :]
        mask = aux_state[:, 0 : self.config.mask_size]  # Dims: batch, len(mask)
        mask = mask < 0.5
        task = tf.reshape(
            aux_state[:, -self.config.task_size :], [-1, 1, self.config.task_size]
        )
        return ego_state, npc_state, mask, task

    def attention(self, query, key, value, mask):
        """
            Compute a Scaled Dot Product Attention.
        :param query: size: batch, head, 1 (ego-entity), features
        :param key:  size: batch, head, entities, features
        :param value: size: batch, head, entities, features
        :param mask: size: batch,  head, 1 (absence feature), 1 (ego-entity)
        :return: the attention softmax(QK^T/sqrt(dk))V
        """
        d_k = self.features_per_head
        scores = tf.matmul(query, tf.transpose(key, perm=[0, 1, 3, 2])) / np.sqrt(d_k)
        mask_constant = scores * 0 + -1e9
        if mask is not None:
            scores = tf.where(mask, mask_constant, scores)
        p_attn = tf.nn.softmax(scores, dim=-1)
        att_output = tf.matmul(p_attn, value)
        return att_output, p_attn

    def build_actor_network(self, name):
        with tf.compat.v1.variable_scope(name):
            state_inputs = tf.compat.v1.placeholder(
                tf.float32, [None, self.all_state_size], name="state_inputs"
            )
            ego_state, npc_state, mask, task = self.split_input(state_inputs)
            # ego
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
            task_encoder_1 = tf.compat.v1.layers.dense(
                inputs=task,
                units=64,
                activation=tf.nn.tanh,
                kernel_initializer=tf.compat.v1.variance_scaling_initializer(),
                name="task_encoder_1",
            )
            task_encoder_2 = tf.compat.v1.layers.dense(
                inputs=task_encoder_1,
                units=64,
                activation=tf.nn.tanh,
                kernel_initializer=tf.compat.v1.variance_scaling_initializer(),
                name="task_encoder_2",
            )
            ego_encoder_3 = tf.concat(
                [ego_encoder_2, task_encoder_2], axis=2, name="ego_encoder_3"
            )  # Dims: batch, 1, 128
            ego_encoder_4 = tf.compat.v1.layers.dense(
                inputs=ego_encoder_3,
                units=64,
                activation=tf.nn.tanh,
                kernel_initializer=tf.compat.v1.variance_scaling_initializer(),
                name="ego_encoder_4",
            )
            # npc
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
            )  # Dims: batch, entities, 64
            all_encoder = tf.concat(
                [ego_encoder_4, npc_encoder_2], axis=1
            )  # Dims: batch, npcs_entities + 1, 64

            # attention layer
            query_ego = tf.compat.v1.layers.dense(
                inputs=ego_encoder_4,
                units=64,
                use_bias=None,
                kernel_initializer=tf.compat.v1.variance_scaling_initializer(),
                name="query_ego",
            )
            key_all = tf.compat.v1.layers.dense(
                inputs=all_encoder,
                units=64,
                use_bias=None,
                kernel_initializer=tf.compat.v1.variance_scaling_initializer(),
                name="key_all",
            )
            value_all = tf.compat.v1.layers.dense(
                inputs=all_encoder,
                units=64,
                use_bias=None,
                kernel_initializer=tf.compat.v1.variance_scaling_initializer(),
                name="value_all",
            )
            # Dimensions: Batch, entity, head, feature_per_head
            query_ego = tf.reshape(
                query_ego, [-1, 1, self.feature_head, self.features_per_head]
            )
            key_all = tf.reshape(
                key_all,
                [
                    -1,
                    self.config.npc_num + 1,
                    self.feature_head,
                    self.features_per_head,
                ],
            )
            value_all = tf.reshape(
                value_all,
                [
                    -1,
                    self.config.npc_num + 1,
                    self.feature_head,
                    self.features_per_head,
                ],
            )
            # Dimensions: Batch, head, entity, feature_per_head，改一下顺序
            query_ego = tf.transpose(query_ego, perm=[0, 2, 1, 3])
            key_all = tf.transpose(key_all, perm=[0, 2, 1, 3])
            value_all = tf.transpose(value_all, perm=[0, 2, 1, 3])
            mask = tf.reshape(mask, [-1, 1, 1, self.config.mask_size])
            mask = tf.tile(mask, [1, self.feature_head, 1, 1])
            # attention mechanism and its outcome
            att_result, att_matrix = self.attention(query_ego, key_all, value_all, mask)
            att_matrix = tf.identity(att_matrix, name="att_matrix")
            att_result = tf.reshape(
                att_result,
                [-1, self.features_per_head * self.feature_head],
                name="att_result",
            )
            att_combine = tf.compat.v1.layers.dense(
                inputs=att_result,
                units=64,
                use_bias=None,
                kernel_initializer=tf.compat.v1.variance_scaling_initializer(),
                name="attention_combine",
            )
            att_with_task = tf.concat(
                [att_combine, tf.squeeze(task_encoder_2, axis=1)],
                axis=1,
                name="att_with_task",
            )

            # action output layer
            action_1 = tf.compat.v1.layers.dense(
                inputs=att_with_task,
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
        return state_inputs, actor_variables, tf.squeeze(action), att_matrix

    def get_attention_matrix(self, sess, state):
        if len(state.shape) < 2:
            state = state.reshape((1, *state.shape))
        attention_matrix = sess.run(
            self.attention_matrix, feed_dict={self.state_inputs: state}
        )
        return attention_matrix

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


class SocMtCriticNetwork:
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

        self.target = tf.compat.v1.placeholder(
            tf.float32, [None, self.config.task_size]
        )
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

    def split_input(self, all_state):
        # state:[batch, ego_feature_num + npc_feature_num*npc_num + mask]
        env_state = all_state[
            :,
            0 : self.config.ego_feature_num
            + self.config.npc_num * self.config.npc_feature_num,
        ]  # Dims: batch, (ego+npcs)features
        ego_state = tf.reshape(
            env_state[:, 0 : self.config.ego_feature_num],
            [-1, 1, self.config.ego_feature_num],
        )  # Dims: batch, 1, features
        npc_state = tf.reshape(
            env_state[:, self.config.ego_feature_num :],
            [-1, self.config.npc_num, self.config.npc_feature_num],
        )  # Dims: batch, entities, features

        aux_state = all_state[:, -(self.config.mask_size + self.config.task_size) :]
        mask = aux_state[:, 0 : self.config.mask_size]  # Dims: batch, len(mask)
        mask = mask < 0.5
        task = aux_state[:, -self.config.task_size :]
        return ego_state, npc_state, mask, task

    def build_critic_network(self, name):
        with tf.compat.v1.variable_scope(name):
            state_inputs = tf.compat.v1.placeholder(
                tf.float32, [None, self.all_state_size], name="state_inputs"
            )
            action_inputs = tf.compat.v1.placeholder(
                tf.float32, [None, self.action_size], name="action_inputs"
            )
            ego_state, npc_state, mask, task = self.split_input(state_inputs)
            ego_state = tf.squeeze(ego_state, axis=1)
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
                [encoder_2[:, i] for i in range(self.config.npc_num)],
                axis=1,
                name="concat",
            )
            # task fc
            task_encoder = tf.compat.v1.layers.dense(
                inputs=task,
                units=64,
                activation=tf.nn.tanh,
                kernel_initializer=tf.compat.v1.variance_scaling_initializer(),
                name="task_encoder",
            )
            # converge
            fc_1 = tf.concat([ego_state, concat, task_encoder], axis=1, name="fc_1")
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
                units=self.config.task_size,
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


def OU(action, mu=0, theta=0.15, sigma=0.3):
    noise = theta * (mu - action) + sigma * np.random.randn(1)
    return noise
