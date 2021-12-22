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


def get_split_batch(batch):
    """memory.sample() returns a batch of experiences, but we want an array
    for each element in the memory (s, a, r, s', done)"""
    states_mb = np.array([each[0][0] for each in batch])
    # print(states_mb.shape)
    actions_mb = np.array([each[0][1] for each in batch])
    # print(actions_mb.shape)
    rewards_mb = np.array([each[0][2] for each in batch])
    # print(rewards_mb.shape)
    next_states_mb = np.array([each[0][3] for each in batch])
    # print(next_states_mb.shape)
    dones_mb = np.array([each[0][4] for each in batch])

    return states_mb, actions_mb, rewards_mb, next_states_mb, dones_mb


def OU(action, mu=0, theta=0.15, sigma=0.3):
    # noise = np.ones(action_dim) * mu
    noise = theta * (mu - action) + sigma * np.random.randn(1)
    # noise = noise + d_noise
    return noise


def calculate_angle(ego_location, goal_location, ego_direction):
    # calculate vector direction
    goal_location = np.array(goal_location)
    ego_location = np.array(ego_location)
    goal_vector = goal_location - ego_location
    L_g_vector = np.sqrt(goal_vector.dot(goal_vector))
    ego_vector = np.array(
        [np.cos(ego_direction * np.pi / 180), np.sin(ego_direction * np.pi / 180)]
    )
    L_e_vector = np.sqrt(ego_vector.dot(ego_vector))
    cos_angle = goal_vector.dot(ego_vector) / (L_g_vector * L_e_vector)
    angle = (np.arccos(cos_angle)) * 180 / np.pi
    if np.cross(goal_vector, ego_vector) > 0:
        angle = -angle
    return angle


def calculate_distance(location_a, location_b):
    """ calculate distance between a and b"""
    return np.linalg.norm(location_a - location_b)
