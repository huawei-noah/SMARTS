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


def q_net(observation_spec, action_spec):
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
        observation_spec(),
        action_spec(),
        preprocessing_layers=preprocessing_layer,
        conv_layer_params=conv_layer_params,
        fc_layer_params=fc_layer_params,
    )
    return q_net
