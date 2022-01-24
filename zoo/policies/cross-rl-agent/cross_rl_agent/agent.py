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
# This file contains an RLlib-trained policy evaluation usage (not for training).

import tensorflow as tf

from smarts.core.agent import Agent

from .cross_space import SocMtActorNetwork, SocMtCriticNetwork


def init_tensorflow():
    configProto = tf.compat.v1.ConfigProto()
    configProto.gpu_options.allow_growth = True
    # reset tensorflow graph
    tf.compat.v1.reset_default_graph()
    return configProto


class RLAgent(Agent):
    def __init__(self, load_path, policy_name):
        configProto = init_tensorflow()
        model_name = policy_name
        self.actor = SocMtActorNetwork(name="actor")
        critic_1 = SocMtCriticNetwork(name="critic_1")
        critic_2 = SocMtCriticNetwork(name="critic_2")
        saver = tf.compat.v1.train.Saver()

        self.sess = tf.compat.v1.Session(config=configProto)

        saver = tf.compat.v1.train.import_meta_graph(
            load_path + model_name + ".ckpt" + ".meta"
        )
        saver.restore(self.sess, load_path + model_name + ".ckpt")
        if saver is None:
            print("did not load")

    def act(self, state):
        action = self.actor.get_action_noise(self.sess, state, rate=-1)
        return action
