import numpy as np
import tensorflow as tf
from tf_agents.networks.q_network import QNetwork


def q_net(env):
    preprocessing_layer_rgb = tf.keras.layers.Lambda(
        lambda obs: tf.cast(obs["rgb"], np.float32) / 255.0
    )
    # number of filters, kernel size, and stride
    conv_layer_params = [(32, (8, 8), 4), (64, (4, 4), 2), (64, (3, 3), 1)]
    fc_layer_params = [512]
    q_net = QNetwork(
        env.observation_spec(),
        env.action_spec(),
        preprocessing_layers=preprocessing_layer_rgb,
        conv_layer_params=conv_layer_params,
        fc_layer_params=fc_layer_params,
    )
    return q_net


# class NormAndStack(tf.keras.layers.Layer):

#     def __init__(self, input_spec, **kwargs):
#         super(NormAndStack, self).__init__(**kwargs)
#         self._input_spec = input_spec

#     def call(self, inputs):  # Defines the computation from inputs to outputs
#         normed = tf.cast(inputs, np.float32) / 255.0
#         stacked = normed
#         print("dddddddddddddddddddddddddddddddddd11111")
#         print(self._input_spec)
#         print(type(inputs))
#         # print(input)
#         print("dddddddddddddddddddddddddddddddddd")
#         return stacked

#     def compute_output_shape(self, input_shape):
#         return input_shape

#     def get_config(self):
#         config = {
#             'input_spec': self._input_spec,
#         }
#         base_config = super(NormAndStack, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))
