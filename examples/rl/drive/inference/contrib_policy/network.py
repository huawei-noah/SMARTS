import gymnasium as gym
import torch as th
import torch.nn as nn
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN
from stable_baselines3.common.type_aliases import TensorDict


class CombinedExtractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Dict, cnn_output_dim: int):
        super(CombinedExtractor, self).__init__(observation_space, features_dim=1)
        # We assume CxHxW images (channels first)

        extractors = {}

        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if key == "rgb":
                extractors[key] = NatureCNN(subspace, features_dim=cnn_output_dim)
                total_concat_size += cnn_output_dim
            else:
                # The observation key is a vector, flatten it if needed
                extractors[key] = nn.Flatten()
                total_concat_size += get_flattened_obs_dim(subspace)

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations: TensorDict) -> th.Tensor:
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        return th.cat(encoded_tensor_list, dim=1)


def combined_extractor(config):
    kwargs = {}
    kwargs["policy"] = "MultiInputPolicy"
    kwargs["policy_kwargs"] = dict(
        features_extractor_class=CombinedExtractor,
        features_extractor_kwargs=dict(cnn_output_dim=256),
        # net_arch=[],
        # net_arch=[dict(pi=[256], vf=[256])],
    )
    kwargs.update(config.get("alg", {}))

    return kwargs
