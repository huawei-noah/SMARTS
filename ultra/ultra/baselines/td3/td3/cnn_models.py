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
import math
from typing import Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1.0 / np.sqrt(fan_in)
    return (-lim, lim)


def _conv2d_output_size(
    input_dimensions: Tuple[int, int],
    output_channels: Sequence[int],
    kernel_sizes: Sequence[int],
    stride_sizes: Sequence[int],
) -> int:
    """Calculates the output size of convolutional layers based on square kernel sizes
    and square stride sizes. This does not take into account padding or dilation.

        H_out = \floor((H_in - kernel_size) / stride_size + 1)
        W_out = \floor((W_in - kernel_size) / stride_size + 1)

    Args:
        input_dimensions (Tuple[int, int]): The input dimension in the form of
            (height, width).
        output_channels (Sequence[int]): Number of output channels for each layer of
            convolutions.
        kernel_sizes (Sequence[int]): The kernel size for each layer of
            convolutions.
        stride_sizes (Sequence[int]): The stride size for each layer of
            convolutions.

    Returns:
        int: The output size of the convolutional layers.
    """
    assert len(output_channels) == len(kernel_sizes) == len(stride_sizes)
    assert len(input_dimensions) == 2

    input_height = input_dimensions[0]
    input_width = input_dimensions[1]

    for kernel_size, stride_size in zip(kernel_sizes, stride_sizes):
        input_height = math.floor((input_height - kernel_size) / stride_size + 1)
        input_width = math.floor((input_width - kernel_size) / stride_size + 1)

    return input_height * input_width * output_channels[-1]


class ActorNetwork(torch.nn.Module):
    CONV_OUTPUT_CHANNELS = [32, 64, 64]
    CONV_KERNEL_SIZES = [8, 4, 3]
    CONV_STRIDE_SIZES = [4, 2, 1]

    def __init__(
        self,
        input_channels,
        input_dimension,
        action_size,
        seed,
        hidden_dim=512,
    ):
        super(ActorNetwork, self).__init__()
        self.action_size = action_size
        self.seed = torch.manual_seed(seed)

        self.conv1 = torch.nn.Conv2d(
            input_channels,
            ActorNetwork.CONV_OUTPUT_CHANNELS[0],
            ActorNetwork.CONV_KERNEL_SIZES[0],
            stride=ActorNetwork.CONV_STRIDE_SIZES[0],
        )
        self.conv2 = torch.nn.Conv2d(
            ActorNetwork.CONV_OUTPUT_CHANNELS[0],
            ActorNetwork.CONV_OUTPUT_CHANNELS[1],
            ActorNetwork.CONV_KERNEL_SIZES[1],
            stride=ActorNetwork.CONV_STRIDE_SIZES[1],
        )
        self.conv3 = torch.nn.Conv2d(
            ActorNetwork.CONV_OUTPUT_CHANNELS[1],
            ActorNetwork.CONV_OUTPUT_CHANNELS[2],
            ActorNetwork.CONV_KERNEL_SIZES[2],
            stride=ActorNetwork.CONV_STRIDE_SIZES[2],
        )
        conv_output_size = _conv2d_output_size(
            input_dimension,
            ActorNetwork.CONV_OUTPUT_CHANNELS,
            ActorNetwork.CONV_KERNEL_SIZES,
            ActorNetwork.CONV_STRIDE_SIZES,
        )
        self.fc1 = torch.nn.Linear(conv_output_size, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.weight.data.uniform_(*hidden_init(self.conv1))
        self.conv2.weight.data.uniform_(*hidden_init(self.conv2))
        self.conv3.weight.data.uniform_(*hidden_init(self.conv3))
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(-1e-3, 1e-3)

    def forward(self, image, training=False):
        output = F.relu(self.conv1(image))
        output = F.relu(self.conv2(output))
        output = F.relu(self.conv3(output))
        output = output.view(output.size(0), -1)
        output = F.relu(self.fc1(output))
        output = self.fc2(output)

        if training:
            aux_losses = {}
            return output, aux_losses
        else:
            return output


class CriticNetwork(torch.nn.Module):
    CONV_OUTPUT_CHANNELS = [32, 64, 64]
    CONV_KERNEL_SIZES = [8, 4, 3]
    CONV_STRIDE_SIZES = [4, 2, 1]

    def __init__(
        self,
        input_channels,
        input_dimension,
        action_size,
        seed,
        hidden_dim=512,
    ):
        super(CriticNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.action_size = action_size

        self.conv1 = torch.nn.Conv2d(
            input_channels,
            CriticNetwork.CONV_OUTPUT_CHANNELS[0],
            CriticNetwork.CONV_KERNEL_SIZES[0],
            stride=CriticNetwork.CONV_STRIDE_SIZES[0],
        )
        self.conv2 = torch.nn.Conv2d(
            CriticNetwork.CONV_OUTPUT_CHANNELS[0],
            CriticNetwork.CONV_OUTPUT_CHANNELS[1],
            CriticNetwork.CONV_KERNEL_SIZES[1],
            stride=CriticNetwork.CONV_STRIDE_SIZES[1],
        )
        self.conv3 = torch.nn.Conv2d(
            CriticNetwork.CONV_OUTPUT_CHANNELS[1],
            CriticNetwork.CONV_OUTPUT_CHANNELS[2],
            CriticNetwork.CONV_KERNEL_SIZES[2],
            stride=CriticNetwork.CONV_STRIDE_SIZES[2],
        )
        conv_output_size = _conv2d_output_size(
            input_dimension,
            CriticNetwork.CONV_OUTPUT_CHANNELS,
            CriticNetwork.CONV_KERNEL_SIZES,
            CriticNetwork.CONV_STRIDE_SIZES,
        )
        self.fc1 = torch.nn.Linear(conv_output_size + action_size, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def reset_parameters(self):
        self.conv1.weight.data.uniform_(*hidden_init(self.conv1))
        self.conv2.weight.data.uniform_(*hidden_init(self.conv2))
        self.conv3.weight.data.uniform_(*hidden_init(self.conv3))
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(-1e-3, 1e-3)

    def forward(self, image, action, training=False):
        output = F.relu(self.conv1(image))
        output = F.relu(self.conv2(output))
        output = F.relu(self.conv3(output))
        output = output.view(output.size(0), -1)
        output = torch.cat((output, action), dim=1)
        output = F.relu(self.fc1(output))
        output = self.fc2(output)

        if training:
            aux_losses = {}
            return output, aux_losses
        else:
            return output
