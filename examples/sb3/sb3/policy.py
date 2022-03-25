import gym
import torch as th
import torch.nn as nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from torchsummary import summary


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=n_input_channels,
                out_channels=48,
                kernel_size=4,
                stride=2,
                padding=0,
            ),
            nn.ELU(),
            nn.Conv2d(
                in_channels=48, out_channels=96, kernel_size=4, stride=2, padding=0
            ),
            nn.ELU(),
            nn.Conv2d(
                in_channels=96, out_channels=192, kernel_size=4, stride=2, padding=0
            ),
            nn.ELU(),
            nn.Conv2d(
                in_channels=192, out_channels=384, kernel_size=4, stride=2, padding=0
            ),
            nn.ELU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ELU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))
