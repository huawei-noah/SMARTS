import gym
import torch
import torch.nn as nn
from sb3 import util as sb3_util
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import get_linear_fn


class Dreamer(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super(Dreamer, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
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
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ELU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


class L5Kit(BaseFeaturesExtractor):
    """Custom feature extractor from raster images for the RL Policy.
    :param observation_space: the input observation space
    :param features_dim: the number of features to extract from the input
    :param model_arch: the model architecture used to extract the features
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super(L5Kit, self).__init__(observation_space, features_dim)

        # We assume CxHxW images (channels first)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(
                n_input_channels,
                64,
                kernel_size=(7, 7),
                stride=(2, 2),
                padding=(3, 3),
                bias=False,
            ),
            nn.GroupNorm(4, 64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                64, 32, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
            ),
            nn.GroupNorm(2, 32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        # nn.Linear(in_features=1568, out_features=features_dim)
        self.linear = nn.Sequential(
            nn.Linear(in_features=n_flatten, out_features=features_dim)
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # sb3_util.plotter3d(observations, rgb_gray=3, name="L5KIT", block=False)
        return self.linear(self.cnn(observations))


class R2plus1D_18(BaseFeaturesExtractor):
    def __init__(
        self, observation_space: gym.spaces.Box, config, pretrained: bool, features_dim: int = 400
    ):
        super().__init__(observation_space, features_dim)
        self._input_channel = 3
        self._input_frames = config["n_stack"]
        self._input_height = config["img_pixels"]
        self._input_width = config["img_pixels"]

        # We assume CxHxW images (channels first)
        assert observation_space.shape == (
            self._input_channel * self._input_frames,
            self._input_height,
            self._input_width,
        )

        import torchvision.models as th_models

        self.thmodel = th_models.video.r2plus1d_18(pretrained=pretrained, progress=True)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # sb3_util.plotter3d(obs, rgb_gray=3, name="R2Plus1D_18", block=False)
        obs = self.modify_obs(obs)
        return self.thmodel(obs)

    def modify_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """
        All pre-trained models expect input images normalized in the
        same way, i.e. mini-batches of 3-channel RGB videos of shape
        (C x F x H x W), where H and W are expected to be 112, and F
        is a number of video frames in a clip. The images have to be
        loaded in to a range of [0, 1].

        Args:
            obs (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """

        # Reshape and swap axes of input image
        obs = torch.reshape(
            obs,
            (
                obs.shape[0],
                self._input_frames,
                self._input_channel,
                self._input_height,
                self._input_width,
            ),
        )
        obs = torch.swapaxes(obs, 1, 2)

        # sb3_util.plotter3d(obs, rgb_gray=3, name="R2plus1D_18")

        return obs


# class NatureCNN(BaseFeaturesExtractor):
#     """
#     CNN from DQN nature paper:
#         Mnih, Volodymyr, et al.
#         "Human-level control through deep reinforcement learning."
#         Nature 518.7540 (2015): 529-533.

#     :param observation_space:
#     :param features_dim: Number of features extracted.
#         This corresponds to the number of unit for the last layer.
#     """

#     def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
#         super(NatureCNN, self).__init__(observation_space, features_dim)
#         # We assume CxHxW images (channels first)
#         # Re-ordering will be done by pre-preprocessing or wrapper
#         assert is_image_space(observation_space, check_channels=False), (
#             "You should use NatureCNN "
#             f"only with images not with {observation_space}\n"
#             "(you are probably using `CnnPolicy` instead of `MlpPolicy` or `MultiInputPolicy`)\n"
#             "If you are using a custom environment,\n"
#             "please check it using our env checker:\n"
#             "https://stable-baselines3.readthedocs.io/en/master/common/env_checker.html"
#         )
#         n_input_channels = observation_space.shape[0]
#         self.cnn = nn.Sequential(
#             nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
#             nn.ReLU(),
#             nn.Flatten(),
#         )

#         # Compute shape by doing one forward pass
#         with th.no_grad():
#             n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

#         self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

#     def forward(self, observations: th.Tensor) -> th.Tensor:
#         return self.linear(self.cnn(observations))


def naturecnn(config):
    kwargs = {}
    kwargs["policy"]="CnnPolicy"
    # kwargs["policy_kwargs"] = dict(
    #     # activation_fn=th.nn.Tanh, # default activation used
    #     net_arch=[],
    # )
    kwargs["target_kl"]=0.003
    # kwargs["ent_coef"]=0.01
    return kwargs


def customnaturecnn(config):
    kwargs = {}
    kwargs["policy"]="CnnPolicy"
    kwargs["policy_kwargs"] = dict(
        # features_extractor_class=NatureCNN,
        # activation_fn=th.nn.Tanh, # default activation used
        net_arch=[128, dict(pi=[32, 32], vf=[32, 32])],
    )
    return kwargs


def dreamer(config):
    kwargs = {}
    kwargs["policy"]="CnnPolicy"
    kwargs["policy_kwargs"] = dict(
        features_extractor_class=Dreamer,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=[],
    )
    return kwargs


def l5kit(config):
    kwargs = {}
    kwargs["policy"]="CnnPolicy"
    kwargs["policy_kwargs"] = dict(
        features_extractor_class=L5Kit,
        features_extractor_kwargs=dict(features_dim=128),
        normalize_images=False,
        net_arch=[],
        log_std_init=-10.0, # reduce exploration
    )

    # Clipping schedule of PPO epsilon parameter
    start_val = 0.1
    end_val = 0.01
    training_progress_ratio = 1.0
    kwargs["clip_range"] = get_linear_fn(start_val, end_val, training_progress_ratio)

    # Hyperparameter
    kwargs["learning_rate"] = 3e-4
    kwargs["n_steps"] = 256
    kwargs["gamma"] = 0.8
    kwargs["gae_lambda"] = 0.9
    kwargs["n_epochs"] = 10
    kwargs["batch_size"] = 64
    kwargs["seed"] = 42

    return kwargs


def r2plus1d_18(config):
    kwargs = {}
    kwargs["policy"]="CnnPolicy"
    kwargs["policy_kwargs"] = dict(
        features_extractor_class=R2plus1D_18,
        features_extractor_kwargs=dict(config=config, pretrained=False, features_dim=400),
        net_arch=[],
        # log_std_init=0.0, # default
        # log_std_init=-10.0, # reduce exploration
    )

    # Hyperparameter
    # kwargs["n_steps"] = 256
    kwargs["batch_size"] = 64

    return kwargs


def dqn_naturecnn(config):
    kwargs = {}
    kwargs["policy"]="CnnPolicy"
    kwargs["buffer_size"]=1_00_000

    return kwargs