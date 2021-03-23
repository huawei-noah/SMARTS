"""
This file contains an RLlib-trained policy evaluation usage (not for training).
"""
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
