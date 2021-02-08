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
import torch
from torch import nn
from torch.distributions.normal import Normal
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFCNet
from ultra.baselines.common.state_preprocessor import *
from ultra.baselines.ppo.ppo.network import PPONetwork


class TorchPPOModel(TorchModelV2, nn.Module):
    """Example of interpreting repeated observations."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        # why num_outputs==6 and it is not configured based on action_space??
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        print("NAME   ", name)
        # self.model = TorchFCNet(
        #     obs_space, action_space, num_outputs, model_config, name
        # )
        self.model = TorchFCNet(
            obs_space, action_space, num_outputs, model_config, name
        )
        self.torchmodel = PPONetwork(
            action_size=model_config["custom_model_config"]["action_size"],
            state_size=model_config["custom_model_config"]["state_size"],
            hidden_units=model_config["custom_model_config"]["hidden_units"],
            init_std=model_config["custom_model_config"]["init_std"],
            seed=model_config["custom_model_config"]["seed"],
            social_feature_encoder_class=model_config["custom_model_config"][
                "social_feature_encoder_class"
            ],
            social_feature_encoder_params=model_config["custom_model_config"][
                "social_feature_encoder_params"
            ],
        )

    def forward(self, input_dict, state, seq_lens):

        print("**** obs", input_dict["obs"].keys())
        dist, value = self.torchmodel(input_dict["obs"])

        # need train/eval?
        action = dist.sample()
        log_prob = dist.log_prob(action)
        action = torch.squeeze(action)
        action = action.data.cpu().numpy()

        action = np.asarray([to_3d_action(a) for a in action])
        print("ACTION", action)
        # return action ,[]
        dummy = self.model.forward(input_dict, state, seq_lens)
        print(dummy)
        print(M)
        # todo why action_Space is 6 d?

    def value_function(self):
        return self.model.value_function()
