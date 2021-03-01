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
# code integrated from
#  1- https://github.com/udacity/deep-reinforcement-learning
#  2- https://github.com/sfujim/TD3/blob/master/TD3.py
#

import numpy as np
import torch
import torch.nn.functional as F


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1.0 / np.sqrt(fan_in)
    return (-lim, lim)


class ActorNetwork(torch.nn.Module):
    def __init__(
        self,
        input_channels,
        low_dim_state_size,
        action_space,
        seed,
        hidden_dim=512,
    ):
        super(ActorNetwork, self).__init__()
        self.state_space = low_dim_state_size
        self.action_space = action_space
        self.seed = torch.manual_seed(seed)

        self.conv1 = torch.nn.Conv2d(
            input_channels, 32, 8, stride=4
        )  # 84-8+4/4 -> 20x20x32
        self.conv2 = torch.nn.Conv2d(32, 64, 4, stride=2)  # 24-3+2/2 -> 9x9x64
        self.conv3 = torch.nn.Conv2d(64, 64, 3, stride=1)  # 9-3+1/1 -> 7x7x64
        self.fc1 = torch.nn.Linear(7 * 7 * 64 + low_dim_state_size, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_space)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.weight.data.uniform_(*hidden_init(self.conv1))
        self.conv2.weight.data.uniform_(*hidden_init(self.conv2))
        self.conv3.weight.data.uniform_(*hidden_init(self.conv3))
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(-1e-3, 1e-3)

    def forward(self, image, low_dim_state):
        output = F.relu(self.conv1(image))
        output = F.relu(self.conv2(output))
        output = F.relu(self.conv3(output))
        output = output.view(output.size(0), -1)
        output = torch.cat((output, low_dim_state), dim=1)
        output = F.relu(self.fc1(output))
        return self.fc2(output)  # remove torch.tanh


class CriticNetwork(torch.nn.Module):
    def __init__(
        self,
        input_channels,
        low_dim_state_size,
        action_space,
        seed,
        hidden_dim=512,
    ):
        super(CriticNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.state_space = low_dim_state_size
        self.action_space = action_space

        self.conv1 = torch.nn.Conv2d(
            input_channels, 32, 8, stride=4
        )  # 100-8+4/4 -> 24x24x32
        self.conv2 = torch.nn.Conv2d(32, 64, 4, stride=2)  # 24-3+2/2 -> 11x11x64
        self.conv3 = torch.nn.Conv2d(64, 64, 3, stride=1)  # 9-3+1/1 -> 9x9x64
        self.fc1 = torch.nn.Linear(
            7 * 7 * 64 + low_dim_state_size + action_space, hidden_dim
        )
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def reset_parameters(self):
        self.conv1.weight.data.uniform_(*hidden_init(self.conv1))
        self.conv2.weight.data.uniform_(*hidden_init(self.conv2))
        self.conv3.weight.data.uniform_(*hidden_init(self.conv3))
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(-1e-3, 1e-3)

    def forward(self, image, low_dim_state, action):
        output = F.relu(self.conv1(image))
        output = F.relu(self.conv2(output))
        output = F.relu(self.conv3(output))
        output = output.view(output.size(0), -1)
        output = torch.cat((output, low_dim_state, action), dim=1)
        output = F.relu(self.fc1(output))
        return self.fc2(output)
