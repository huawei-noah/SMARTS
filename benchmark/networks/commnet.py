import numpy as np
import gym

from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.annotations import override
from ray.rllib.utils.types import ModelConfigDict, TensorType, List, Dict
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.tf.tf_modelv2 import TFModelV2

# import tensorflow as tf1

tf1, tf, tfv = try_import_tf()


class CommNet(TFModelV2):
    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
    ):
        super(CommNet, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )

        # assert isinstance(obs_space, gym.spaces.Box) and isinstance(
        #     action_space, gym.spaces.Tuple
        # ), (obs_space, action_space)
        custom_model_config = model_config["custom_model_config"]

        # TODO(ming): transfer something to here
        self.agent_num = len(obs_space.original_space.spaces)
        self.communicate_level = custom_model_config["communicate_level"]
        self.rnn_hidden_dim = custom_model_config["rnn_hidden_dim"]

        self.encoding = tf1.keras.layers.Dense(
            self.rnn_hidden_dim,
            input_shape=(None, obs_space.shape[0] // self.agent_num),
        )
        self.encoding.build((self.obs_space.shape[0] // self.agent_num,))
        self.f_obs = tf1.keras.layers.GRUCell(self.rnn_hidden_dim)
        self.f_obs.build((self.rnn_hidden_dim,))
        self.f_comm = tf1.keras.layers.GRUCell(self.rnn_hidden_dim)
        self.f_comm.build((self.rnn_hidden_dim,))
        self.decoding = tf1.keras.layers.Dense(
            num_outputs // self.agent_num,
            input_shape=(None, self.rnn_hidden_dim),
            name="logits_out",
            activation=None,
            kernel_initializer=normc_initializer(0.01),
        )
        self.decoding.build((self.rnn_hidden_dim,))

        self.register_variables(self.encoding.variables)
        self.register_variables(self.f_obs.variables)
        self.register_variables(self.f_comm.variables)
        self.register_variables(self.decoding.variables)

    @override(ModelV2)
    def get_initial_state(self) -> List[np.ndarray]:
        return [np.zeros((1, self.rnn_hidden_dim), np.float32)]

    @override(ModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):
        obs_flat = input_dict["obs_flat"]

        obs_flat = tf.reshape(obs_flat, (-1, self.obs_space.shape[0] // self.agent_num))
        obs_encoding = tf.nn.sigmoid(self.encoding(obs_flat))
        obs_encoding = tf.expand_dims(obs_encoding, axis=1)
        h_out, _ = self.f_obs(obs_encoding, state[0])

        for k in range(self.communicate_level):
            if k == 0:
                h = h_out
                c = tf.zeros_like(h)
            else:
                h = tf.reshape(h, (-1, self.agent_num, self.rnn_hidden_dim))
                c = tf.reshape(h, (-1, 1, self.agent_num * self.rnn_hidden_dim))
                c = tf.tile(c, [1, self.agent_num, 1])
                mask = 1.0 - tf.eye(self.agent_num)
                mask = tf.reshape(mask, (-1, 1))
                mask = tf.tile(mask, [1, self.rnn_hidden_dim])
                mask = tf.reshape(mask, (self.agent_num, -1))
                c = c * tf.expand_dims(mask, 0)
                # 因为现在所有agent的h都在最后一维，不能直接加。所以先扩展一维，相加后再去掉
                c = tf.reshape(
                    c, (-1, self.agent_num, self.agent_num, self.rnn_hidden_dim)
                )
                c = tf.reduce_mean(
                    c, axis=-2
                )  # (episode_num * max_episode_len, n_agents, rnn_hidden_dim)
                h = tf.reshape(h, (-1, 1, self.rnn_hidden_dim))
                c = tf.reshape(c, (-1, 1, self.rnn_hidden_dim))

            h, _ = self.f_comm(c, h)
        h = tf.squeeze(h, axis=1)
        weights = self.decoding(h)
        # reshape to every agents
        weights = tf.reshape(weights, (-1, self.num_outputs))
        return weights, [h_out]

    @override(ModelV2)
    def value_function(self) -> TensorType:
        raise NotImplementedError


ModelCatalog.register_custom_model("CommNet", CommNet)
