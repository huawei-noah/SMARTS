"""
General networks for pytorch.

Algorithm-specific networks should go else-where.
"""
import math

import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn import BatchNorm1d

from rlkit.policies.base import Policy
from rlkit.torch.utils import pytorch_util as ptu
from rlkit.torch.core import PyTorchModule
from rlkit.torch.utils.normalizer import TorchFixedNormalizer
from rlkit.torch.common.modules import LayerNorm


def identity(x):
    return x


class Mlp(PyTorchModule):
    def __init__(
        self,
        hidden_sizes,
        output_size,
        input_size,
        init_w=3e-3,
        hidden_activation=F.relu,
        output_activation=identity,
        hidden_init=ptu.fanin_init,
        b_init_value=0.1,
        layer_norm=False,
        layer_norm_kwargs=None,
        batch_norm=False,
        # batch_norm_kwargs=None,
        batch_norm_before_output_activation=False,
    ):
        self.save_init_params(locals())
        super().__init__()

        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.batch_norm_before_output_activation = batch_norm_before_output_activation
        self.fcs = []
        self.layer_norms = []
        self.batch_norms = []
        in_size = input_size

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

            if self.layer_norm:
                ln = LayerNorm(next_size)
                self.__setattr__("layer_norm{}".format(i), ln)
                self.layer_norms.append(ln)

            if self.batch_norm:
                bn = BatchNorm1d(next_size)
                self.__setattr__("batch_norm{}".format(i), bn)
                self.batch_norms.append(bn)

        if self.batch_norm_before_output_activation:
            bn = BatchNorm1d(output_size)
            self.__setattr__("batch_norm_last", bn)
            self.batch_norms.append(bn)

        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, input, return_preactivations=False):
        h = input
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            if self.layer_norm:
                h = self.layer_norms[i](h)
            if self.batch_norm:
                h = self.batch_norms[i](h)
            h = self.hidden_activation(h)
        preactivation = self.last_fc(h)
        if self.batch_norm_before_output_activation:
            preactivation = self.batch_norms[-1](preactivation)
        output = self.output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        else:
            return output


class ConvNet(PyTorchModule):
    def __init__(
        self,
        kernel_sizes,
        num_channels,
        strides,
        paddings,
        hidden_sizes,
        output_size,
        input_size,
        init_w=3e-3,
        hidden_activation=F.relu,
        output_activation=identity,
        hidden_init=ptu.fanin_init,
        b_init_value=0.1,
    ):
        self.save_init_params(locals())
        super().__init__()

        self.kernel_sizes = kernel_sizes
        self.num_channels = num_channels
        self.strides = strides
        self.paddings = paddings
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.convs = []
        self.fcs = []

        in_c = input_size[0]
        in_h = input_size[1]
        for k, c, s, p in zip(kernel_sizes, num_channels, strides, paddings):
            conv = nn.Conv2d(in_c, c, k, stride=s, padding=p)
            hidden_init(conv.weight)
            conv.bias.data.fill_(b_init_value)
            self.convs.append(conv)

            out_h = int(math.floor(1 + (in_h + 2 * p - k) / s))

            in_c = c
            in_h = out_h

        in_dim = in_c * in_h * in_h
        for h in hidden_sizes:
            fc = nn.Linear(in_dim, h)
            in_dim = h
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.fcs.append(fc)

        self.last_fc = nn.Linear(in_dim, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, input, return_preactivations=False):
        h = input
        for conv in self.convs:
            h = conv(h)
            h = self.hidden_activation(h)
        h = h.view(h.size(0), -1)
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            h = self.hidden_activation(h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)
        if return_preactivations:
            return output, preactivation
        else:
            return output


class FlattenMlp(Mlp):
    """
    Flatten inputs along dimension 1 and then pass through MLP.
    """

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=1)
        return super().forward(flat_inputs, **kwargs)


class MlpPolicy(Mlp, Policy):
    """
    A simpler interface for creating policies.
    """

    def __init__(
        self,
        hidden_sizes,
        output_size,
        input_size,
        obs_normalizer: TorchFixedNormalizer = None,
        **kwargs
    ):
        self.save_init_params(locals())
        super().__init__(hidden_sizes, output_size, input_size, **kwargs)
        self.obs_normalizer = obs_normalizer

    def forward(self, obs, **kwargs):
        if self.obs_normalizer:
            obs = self.obs_normalizer.normalize(obs)
        return super().forward(obs, **kwargs)

    def get_action(self, obs_np):
        actions = self.get_actions(obs_np[None])
        return actions[0, :], {}

    def get_actions(self, obs):
        return self.eval_np(obs)


class TanhMlpPolicy(MlpPolicy):
    """
    A helper class since most policies have a tanh output activation.
    """

    def __init__(self, *args, **kwargs):
        self.save_init_params(locals())
        super().__init__(*args, output_activation=torch.tanh, **kwargs)
        raise NotImplementedError()


class ObsPreprocessedQFunc(FlattenMlp):
    """
    This is a weird thing and I didn't know what to call.
    Basically I wanted this so that if you need to preprocess
    your inputs somehow (attention, gating, etc.) with an external module
    before passing to the policy you could do so.
    Assumption is that you do not want to update the parameters of the preprocessing
    module so its output is always detached.
    """

    def __init__(self, preprocess_model, z_dim, *args, wrap_absorbing=False, **kwargs):
        self.save_init_params(locals())
        super().__init__(*args, **kwargs)
        # this is a hack so that it is not added as a submodule
        self.preprocess_model_list = [preprocess_model]
        self.wrap_absorbing = wrap_absorbing
        self.z_dim = z_dim

    @property
    def preprocess_model(self):
        # this is a hack so that it is not added as a submodule
        return self.preprocess_model_list[0]

    def preprocess_fn(self, obs_batch):
        mode = self.preprocess_model.training
        self.preprocess_model.eval()
        processed_obs_batch = self.preprocess_model(
            obs_batch[:, : -self.z_dim],
            self.wrap_absorbing,
            obs_batch[:, -self.z_dim :],
        ).detach()
        self.preprocess_model.train(mode)
        return processed_obs_batch

    def forward(self, obs, actions):
        obs = self.preprocess_fn(obs).detach()
        return super().forward(obs, actions)


class ObsPreprocessedVFunc(FlattenMlp):
    """
    This is a weird thing and I didn't know what to call.
    Basically I wanted this so that if you need to preprocess
    your inputs somehow (attention, gating, etc.) with an external module
    before passing to the policy you could do so.
    Assumption is that you do not want to update the parameters of the preprocessing
    module so its output is always detached.
    """

    def __init__(self, preprocess_model, z_dim, *args, wrap_absorbing=False, **kwargs):
        self.save_init_params(locals())
        super().__init__(*args, **kwargs)
        # this is a hack so that it is not added as a submodule
        self.preprocess_model_list = [preprocess_model]
        self.wrap_absorbing = wrap_absorbing
        self.z_dim = z_dim

    @property
    def preprocess_model(self):
        # this is a hack so that it is not added as a submodule
        return self.preprocess_model_list[0]

    def preprocess_fn(self, obs_batch):
        mode = self.preprocess_model.training
        self.preprocess_model.eval()
        processed_obs_batch = self.preprocess_model(
            obs_batch[:, : -self.z_dim],
            self.wrap_absorbing,
            obs_batch[:, -self.z_dim :],
        ).detach()
        self.preprocess_model.train(mode)
        return processed_obs_batch

    def forward(self, obs):
        obs = self.preprocess_fn(obs).detach()
        return super().forward(obs)
