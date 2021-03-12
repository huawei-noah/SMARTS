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
from collections import OrderedDict
from typing import Dict, List

import numpy as np
import tensorflow as tf
from gym import spaces
from ray.rllib.models import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import get_activation_fn, try_import_tf
from ray.rllib.utils.types import ModelConfigDict, TensorType

tf1, tf, tf_version = try_import_tf()


class CentralizedActorCriticModel(TFModelV2):
    CRITIC_OBS = "critic_obs"

    def __init__(
        self,
        obs_space: spaces.Space,
        action_space: spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
    ):
        super(CentralizedActorCriticModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )

        model_config = model_config["custom_model_config"]
        self.n_agents = model_config["agent_number"]
        if model_config["critic_mode"] == "mean":
            self.critic_obs = spaces.Dict(
                OrderedDict(
                    {
                        "own_obs": self.obs_space,
                        "own_act": self.action_space,
                        "oppo_act": self.action_space,
                    }
                )
            )
        else:
            self.critic_obs = spaces.Dict(
                OrderedDict(
                    {
                        **{f"AGENT-{i}": self.obs_space for i in range(self.n_agents)},
                        **{
                            f"AGENT-{i}-action": self.action_space
                            for i in range(self.n_agents)
                        },
                    }
                )
            )
        self.critic_preprocessor = get_preprocessor(self.critic_obs)(self.critic_obs)
        self.obs_preprocessor = get_preprocessor(self.obs_space)(self.obs_space)
        self.act_preprocessor = get_preprocessor(self.action_space)(self.action_space)

        self.action_model = self._build_action_model(model_config["action_model"])
        self.value_model = self._build_value_model(model_config["value_model"])
        self.register_variables(self.action_model.variables)
        self.register_variables(self.value_model.variables)

    def _build_action_model(self, model_config: ModelConfigDict):
        """Build action model with model configuration
        model_config = {'activation': str, 'hiddens': Sequence}
        """
        activation = get_activation_fn(model_config.get("activation"))
        hiddens = model_config.get("hiddens", [])
        inputs = tf.keras.layers.Input(
            shape=(np.product(self.obs_preprocessor.shape),), name="policy-inputs"
        )

        last_layer = inputs
        for i, size in enumerate(hiddens):
            last_layer = tf.keras.layers.Dense(
                size,
                name="fc_{}".format(i),
                activation=activation,
                kernel_initializer=normc_initializer(1.0),
            )(last_layer)

        logits_out = tf.keras.layers.Dense(
            self.num_outputs,
            name="logits_out",
            activation=None,
            kernel_initializer=normc_initializer(0.01),
        )(last_layer)

        return tf.keras.Model(inputs, [logits_out])

    def _build_value_model(self, model_config: ModelConfigDict):
        """Build value model with given model configuration
        model_config = {'activation': str, 'hiddens': Sequence}
        """
        activation = get_activation_fn(model_config.get("activation"))
        hiddens = model_config.get("hiddens", [])
        inputs = tf.keras.layers.Input(
            shape=(np.product(self.critic_preprocessor.shape),), name="value-inputs"
        )

        last_layer = inputs
        for i, size in enumerate(hiddens):
            last_layer = tf.keras.layers.Dense(
                size,
                name="fc_{}".format(i),
                activation=activation,
                kernel_initializer=normc_initializer(1.0),
            )(last_layer)

        value_out = tf.keras.layers.Dense(
            1,
            name="value_out",
            activation=None,
            kernel_initializer=normc_initializer(0.01),
        )(last_layer)

        return tf.keras.Model(inputs, [value_out])

    @override(ModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):
        # obs = self.obs_preprocessor.transform(input_dict["obs"])
        logits_out = self.action_model(input_dict["obs_flat"])
        return logits_out, state

    def central_value_function(self, critic_obs):
        # Dict({obs, action})
        # critic_obs = self.critic_preprocessor.transform(critic_obs)
        self._value_out = self.value_model(critic_obs)
        return tf.reshape(self._value_out, [-1])

    @override(ModelV2)
    def value_function(self) -> TensorType:
        return tf.reshape(self._value_out, [-1])


ModelCatalog.register_custom_model("CAC", CentralizedActorCriticModel)
