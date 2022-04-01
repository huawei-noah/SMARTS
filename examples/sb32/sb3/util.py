import torch as th
import torch.nn as nn
from torchinfo import summary

def print_model(model, env):
    # Print model summary
    print("\n\n")
    network = Network(model.policy.features_extractor, model.policy.mlp_extractor)
    print(network)
    summary(network, (1,) + env.observation_space.shape)
    print("\n\n")


class Network(nn.Module):
    def __init__(self, feature_extractor: nn.Module, mlp_extractor: nn.Module):
        super(Network, self).__init__()
        self._feature_extractor = feature_extractor
        self._mlp_extractor = mlp_extractor

    def forward(self, obs: th.Tensor) -> th.Tensor:
        feature_out = self._feature_extractor(obs)
        return self._mlp_extractor(feature_out)
