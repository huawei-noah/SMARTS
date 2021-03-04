# MIT License
#
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
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
import torch, gym
from torch import nn
from torch.distributions.normal import Normal
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFCNet
from ultra.baselines.common.state_preprocessor import *


class CustomFCModel(TorchModelV2, nn.Module):
    """Example of interpreting repeated observations."""

    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config,
        name: str,
        **customized_model_kwargs
    ):
        super(CustomFCModel, self).__init__(
            obs_space=obs_space,
            action_space=action_space,
            num_outputs=num_outputs,
            model_config=model_config,
            name=name,
        )
        nn.Module.__init__(self)

        if "adapter" in model_config["custom_model_config"]:
            adapter = model_config["custom_model_config"]["adapter"]
        else:
            adapter = customized_model_kwargs["adapter"]

        social_feature_encoder_class = adapter.social_feature_encoder_class
        social_feature_encoder_params = adapter.social_feature_encoder_params
        self.social_feature_encoder = (
            social_feature_encoder_class(**social_feature_encoder_params)
            if social_feature_encoder_class
            else None
        )

        self.model = TorchFCNet(
            obs_space, action_space, num_outputs, model_config, name
        )

    def forward(self, input_dict, state, seq_lens):

        low_dim_state = input_dict["obs"]["low_dim_states"]
        social_vehicles_state = input_dict["obs"]["social_vehicles"]

        social_feature = []
        if self.social_feature_encoder is not None:
            social_feature, _ = self.social_feature_encoder(social_vehicles_state)
        else:
            social_feature = [e.reshape(1, -1) for e in social_vehicles_state]

        input_dict["obs"]["social_vehicles"] = (
            torch.cat(social_feature, 0) if len(social_feature) > 0 else []
        )
        return self.model.forward(input_dict, state, seq_lens)

    def value_function(self):
        return self.model.value_function()
