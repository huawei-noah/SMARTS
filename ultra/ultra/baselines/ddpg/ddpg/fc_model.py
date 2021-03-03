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
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim


def init_fanin(tensor):
    fanin = tensor.size(1)
    v = 1.0 / np.sqrt(fanin)
    init.uniform_(tensor, -v, v)


def weights_init_(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class ActorNetwork(nn.Module):
    def __init__(self, state_space, action_space, seed, social_feature_encoder=None):
        super(ActorNetwork, self).__init__()
        self.social_feature_encoder = social_feature_encoder
        self.seed = torch.manual_seed(seed)

        self.l1 = nn.Linear(state_space, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, action_space)
        self.reset_parameters()

    def reset_parameters(self):
        init_fanin(self.l1.weight)
        init_fanin(self.l2.weight)
        init.uniform_(self.l3.weight, -3e-3, 3e-3)

    def forward(self, state, training=False):
        low_dim_state = state["low_dim_states"]
        social_vehicles_state = state["social_vehicles"]

        aux_losses = {}
        social_feature = []
        if self.social_feature_encoder is not None:
            social_feature, social_encoder_aux_losses = self.social_feature_encoder(
                social_vehicles_state, training
            )
            aux_losses.update(social_encoder_aux_losses)
        else:
            social_feature = [e.reshape(1, -1) for e in social_vehicles_state]

        social_feature = torch.cat(social_feature, 0) if len(social_feature) > 0 else []
        state = (
            torch.cat([low_dim_state, social_feature], -1)
            if len(social_feature) > 0
            else low_dim_state
        )
        # print('*** ACTOR STATE SIZE', state.shape)
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = torch.tanh(self.l3(a))
        if training:
            return a, aux_losses
        else:
            return a


class CriticNetwork(nn.Module):
    def __init__(self, state_space, action_space, seed, social_feature_encoder=None):
        super(CriticNetwork, self).__init__()
        self.social_feature_encoder = social_feature_encoder
        self.seed = torch.manual_seed(seed)

        self.l1 = nn.Linear(state_space, 64)
        self.l2 = nn.Linear(64 + action_space, 64)
        self.l3 = nn.Linear(64, 1)
        self.reset_parameters()

    def reset_parameters(self):
        init_fanin(self.l1.weight)
        init_fanin(self.l2.weight)
        init_fanin(self.l3.weight)

    def forward(self, state, action, training=False):
        low_dim_state = state["low_dim_states"]
        social_vehicles_state = state["social_vehicles"]

        aux_losses = {}
        social_feature = []
        if self.social_feature_encoder is not None:
            social_feature, social_encoder_aux_losses = self.social_feature_encoder(
                social_vehicles_state, training
            )
            aux_losses.update(social_encoder_aux_losses)
        else:
            social_feature = [e.reshape(1, -1) for e in social_vehicles_state]

        social_feature = torch.cat(social_feature, 0) if len(social_feature) > 0 else []
        state = (
            torch.cat([low_dim_state, social_feature], -1)
            if len(social_feature) > 0
            else low_dim_state
        )
        # print('*** CRITIC STATE SIZE', state.shape)
        q = F.relu(self.l1(state))
        q = F.relu(self.l2(torch.cat([q, action], 1)))
        q = self.l3(q)
        if training:
            return q, aux_losses
        else:
            return q
