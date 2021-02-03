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


class TorchPPOModel(TorchModelV2, nn.Module):
    """Example of interpreting repeated observations."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        self.model = TorchFCNet(
            obs_space, action_space, num_outputs, model_config, name
        )

    def forward(self, input_dict, state, seq_lens):
        # The unpacked input tensors, where M=MAX_PLAYERS, N=MAX_ITEMS:
        # {
        #   'items', <torch.Tensor shape=(?, M, N, 5)>,
        #   'location', <torch.Tensor shape=(?, M, 2)>,
        #   'status', <torch.Tensor shape=(?, M, 10)>,
        # }
        print("The unpacked input tensors:", input_dict["obs"])
        print()
        print("Unbatched repeat dim", input_dict["obs"].unbatch_repeat_dim())
        print()
        print("Fully unbatched", input_dict["obs"].unbatch_all())
        print()
        return self.model.forward(input_dict, state, seq_lens)

    def value_function(self):
        return self.model.value_function()
