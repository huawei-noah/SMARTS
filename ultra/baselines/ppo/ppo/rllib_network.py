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


class TorchPPOModel(TorchModelV2, nn.Module):
    """Example of interpreting repeated observations."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        #why num_outputs==6 and it is not configured based on action_space??
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        # self.model = TorchFCNet(
        #     obs_space, action_space, num_outputs, model_config, name
        # )
        self.model = TorchFCNet(
            obs_space, action_space, num_outputs, model_config, name
        )
        self.torchmodel = nn.Sequential(
            nn.Linear(model_config['custom_model_config']['state_size'], model_config['custom_model_config']['hidden_units']),
            nn.ReLU(),
            nn.Linear(model_config['custom_model_config']['hidden_units'], model_config['custom_model_config']['hidden_units']),
            nn.ReLU(),
            nn.Linear(model_config['custom_model_config']['hidden_units'], model_config['custom_model_config']['action_size']),
            nn.Tanh(),
        )
        # self.state_description = model_config['custom_model_config']['state_description']
        # self.state_preprocessor = StatePreprocessor(
        #     preprocess_state, to_2d_action, self.state_description
        # )
        print('-------------------------')
        # self.social_feature_encoder = model_config['custom_model_config']['social_feature_encoder_class']
        # self.social_capacity = model_config['custom_model_config']['social_capacity']
        # self.social_vehicle_config = model_config['custom_model_config']['social_vehicle_config']
        # self.prev_action = np.zeros(model_config['custom_model_config']['action_size'])
        # self.observation_num_lookahead = model_config['custom_model_config']['observation_num_lookahead']
        # # print('*** Model config', model_config)
        # print('*** num_outputs', num_outputs)
        # print('*** action_space', action_space)
        # print('*** obs_space', obs_space)

    def forward(self, input_dict, state, seq_lens):
        # print('****',input_dict['obs'].keys())
        # print('>>>>', input_dict['obs']['angle_error'].shape)
        # print('<<<<<', len(input_dict['obs']['social_vehicles']), input_dict['obs']['social_vehicles'][0])

        # print("The unpacked input tensors:", input_dict["obs"])

        print('EGO POSITION',len(input_dict['obs']['ego_position']))
        # state = self.state_preprocessor(state=input_dict['obs'],
        #     normalize=True,
        #     device='cpu',
        #     social_capacity=self.social_capacity,
        #     observation_num_lookahead=self.observation_num_lookahead,
        #     social_vehicle_config=self.social_vehicle_config,
        #     prev_action=self.prev_action
        # )
        # # return self.model.forward(input_dict, state, seq_lens)
        # low_dim_state = state["low_dim_states"]
        # social_vehicles_state = state["social_vehicles"]
        #
        # print(state)
        # print('********')
        # aux_losses = {}
        # social_feature = []
        # if self.social_feature_encoder is not None:
        #     social_feature, social_encoder_aux_losses = self.social_feature_encoder(
        #         social_vehicles_state, training
        #     )
        #     aux_losses.update(social_encoder_aux_losses)
        # else:
        #     social_feature = [e.reshape(1, -1) for e in social_vehicles_state]
        # social_feature = torch.cat(social_feature, 0) if len(social_feature) > 0 else []
        # state = (
        #     torch.cat([low_dim_state, social_feature], -1)
        #     if len(social_feature) > 0
        #     else low_dim_state
        # )
        # a = self.model(state)
        # self.prev_action = a
        # if training:
        #     return a, aux_losses
        # else:
        #     return a, {}
        action = self.model.forward(input_dict, state, seq_lens)
        print('ACTION', action)
        return action

    def value_function(self):
        return self.model.value_function()
