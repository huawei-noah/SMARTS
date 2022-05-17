import numpy as np
import tensorflow as tf
from tf_agents.networks.q_network import QNetwork


def q_net(env):
    preprocessing_layer = tf.keras.layers.Lambda(
        lambda obs: tf.cast(obs, np.float32) / 255.0
    )
    # number of filters, kernel size, and stride
    conv_layer_params = [
        (32, (8, 8), 4), 
        (64, (4, 4), 2), 
        (64, (3, 3), 1)
    ]
    fc_layer_params = [512]
    q_net = QNetwork(
        env.observation_spec(),
        env.action_spec(),
        preprocessing_layers=preprocessing_layer,
        conv_layer_params=conv_layer_params,
        fc_layer_params=fc_layer_params,
    )
    return q_net
