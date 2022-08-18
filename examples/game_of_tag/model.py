import torch, gym
from torch import nn
from torch.distributions.normal import Normal
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFCNet


class CustomFCModel(TorchModelV2, nn.Module):
    """Example of interpreting repeated observations."""

    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config,
        name: str,
    ):
        super(CustomFCModel, self).__init__(
            obs_space=obs_space,
            action_space=action_space,
            num_outputs=num_outputs,
            model_config=model_config,
            name=name,
        )
        nn.Module.__init__(self)

        self.model = TorchFCNet(
            obs_space, action_space, num_outputs, model_config, name
        )

    def forward(self, input_dict, state, seq_lens):

        return self.model.forward(input_dict, state, seq_lens)

    def value_function(self):
        return self.model.value_function()
