import numpy as np
import tensorflow as tf
from tf_agents.networks.q_network import QNetwork
from tf_agents.specs import tensor_spec


class CNNNetwork(network.Network):
    def __init__(
        self,
        observation_spec,
        action_spec,
        preprocessing_layers=None,
        preprocessing_combiner=None,
        conv_layer_params=None,
        fc_layer_params=(75, 40),
        dropout_layer_params=None,
        activation_fn=tf.keras.activations.relu,
        enable_last_layer_zero_initializer=False,
        name="ActorNetwork",
    ):
        super(ActorNetwork, self).__init__(
            input_tensor_spec=observation_spec, state_spec=(), name=name
        )

        # For simplicity we will only support a single action float output.
        self._action_spec = action_spec
        flat_action_spec = tf.nest.flatten(action_spec)
        if len(flat_action_spec) > 1:
            raise ValueError("Only a single action is supported by this network")
        self._single_action_spec = flat_action_spec[0]
        if self._single_action_spec.dtype not in [tf.float32, tf.float64]:
            raise ValueError("Only float actions are supported by this network.")

        kernel_initializer = tf.keras.initializers.VarianceScaling(
            scale=1.0 / 3.0, mode="fan_in", distribution="uniform"
        )
        self._encoder = encoding_network.EncodingNetwork(
            observation_spec,
            preprocessing_layers=preprocessing_layers,
            preprocessing_combiner=preprocessing_combiner,
            conv_layer_params=conv_layer_params,
            fc_layer_params=fc_layer_params,
            dropout_layer_params=dropout_layer_params,
            activation_fn=activation_fn,
            kernel_initializer=kernel_initializer,
            batch_squash=False,
        )

        initializer = tf.keras.initializers.RandomUniform(minval=-0.003, maxval=0.003)

        self._action_projection_layer = tf.keras.layers.Dense(
            flat_action_spec[0].shape.num_elements(),
            activation=tf.keras.activations.tanh,
            kernel_initializer=initializer,
            name="action",
        )

    def call(self, observations, step_type=(), network_state=()):
        outer_rank = nest_utils.get_outer_rank(observations, self.input_tensor_spec)
        # We use batch_squash here in case the observations have a time sequence
        # compoment.
        batch_squash = utils.BatchSquash(outer_rank)
        observations = tf.nest.map_structure(batch_squash.flatten, observations)

        state, network_state = self._encoder(
            observations, step_type=step_type, network_state=network_state
        )
        actions = self._action_projection_layer(state)
        actions = common_utils.scale_to_spec(actions, self._single_action_spec)
        actions = batch_squash.unflatten(actions)
        return tf.nest.pack_sequence_as(self._action_spec, [actions]), network_state


def q_net():
    preprocessing_layer = tf.keras.layers.Lambda(
        lambda obs: tf.cast(obs, np.float32) / 255.0
    )
    conv_layer_params = [(32, (8, 8), 4), (64, (4, 4), 2), (64, (3, 3), 1)]
    fc_layer_params = [512]
    q_net = QNetwork(
        tf_env.observation_spec(),
        tf_env.action_spec(),
        preprocessing_layers=preprocessing_layer,
        conv_layer_params=conv_layer_params,
        fc_layer_params=fc_layer_params,
    )

    fc_layer_params = (100, 50)
    action_tensor_spec = tensor_spec.from_spec(env.action_spec())
    num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1

    # Define a helper function to create Dense layers configured with the right
    # activation and kernel initializer.
    def dense_layer(num_units):
        return tf.keras.layers.Dense(
            num_units,
            activation=tf.keras.activations.relu,
            kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=2.0, mode="fan_in", distribution="truncated_normal"
            ),
        )

    preprocessing_layer = keras.layers.Lambda(
        lambda obs: tf.cast(obs, np.float32) / 255.0
    )
    conv_layer_params = [(32, (8, 8), 4), (64, (4, 4), 2), (64, (3, 3), 1)]
    fc_layer_params = [512]

    # QNetwork consists of a sequence of Dense layers followed by a dense layer
    # with `num_actions` units to generate one q_value per available action as
    # its output.
    dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]
    q_values_layer = tf.keras.layers.Dense(
        num_actions,
        activation=None,
        kernel_initializer=tf.keras.initializers.RandomUniform(
            minval=-0.03, maxval=0.03
        ),
        bias_initializer=tf.keras.initializers.Constant(-0.2),
    )
    q_net = sequential.Sequential(dense_layers + [q_values_layer])

    return q_net


# class R2plus1D_18(BaseFeaturesExtractor):
#     def __init__(
#         self,
#         observation_space: gym.spaces.Box,
#         config,
#         pretrained: bool,
#         features_dim: int = 400,
#     ):
#         super().__init__(observation_space, features_dim)
#         self._input_channel = 3
#         self._input_frames = config["n_stack"]
#         self._input_height = config["img_pixels"]
#         self._input_width = config["img_pixels"]

#         # We assume CxHxW images (channels first)
#         assert observation_space.shape == (
#             self._input_channel * self._input_frames,
#             self._input_height,
#             self._input_width,
#         )

#         import torchvision.models as th_models

#         self.thmodel = th_models.video.r2plus1d_18(pretrained=pretrained, progress=True)

#     def forward(self, obs: th.Tensor) -> th.Tensor:
#         # intersection_util.plotter3d(obs, rgb_gray=3, name="R2Plus1D_18", block=False)
#         obs = self.modify_obs(obs)
#         return self.thmodel(obs)

#     def modify_obs(self, obs: th.Tensor) -> th.Tensor:
#         """
#         All pre-trained models expect input images normalized in the
#         same way, i.e. mini-batches of 3-channel RGB videos of shape
#         (C x F x H x W), where H and W are expected to be 112, and F
#         is a number of video frames in a clip. The images have to be
#         loaded in to a range of [0, 1].

#         Args:
#             obs (th.Tensor): _description_

#         Returns:
#             th.Tensor: _description_
#         """

#         # Reshape and swap axes of input image
#         obs = th.reshape(
#             obs,
#             (
#                 obs.shape[0],
#                 self._input_frames,
#                 self._input_channel,
#                 self._input_height,
#                 self._input_width,
#             ),
#         )
#         obs = th.swapaxes(obs, 1, 2)

#         # intersection_util.plotter3d(obs, rgb_gray=3, name="R2plus1D_18")

#         return obs
