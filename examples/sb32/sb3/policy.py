import time
import numpy as np
import gym
import torch
import torch.nn as nn
import torchvision.models as th_models
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

        print("OBSERVATION SPACE INSIDE L5KIT", observation_space)
        sample = observation_space.sample()[None]
        plotter3(sample, rgb_gray=3, name="L5KIT INIT")

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
        print("GOING TO PLOT FROM L5KIT")
        plotter3(observations, rgb_gray=3, name="L5KIT FORWARD")

        return self.linear(self.cnn(observations))


# class R2plus1D_18(BaseFeaturesExtractor):
#     def __init__(self, observation_space: gym.spaces.Box, config, features_dim: int = 400):
#         super().__init__(observation_space, features_dim)

#         # We assume CxHxW images (channels first)
#         space = observation_space.shape
#         assert( space == ( 3*config["n_stack"], config["img_pixels"], config["img_pixels"]) )
#         self.des_shape = (3, config["n_stack"], config["img_pixels"], config["img_pixels"])

#         from torchinfo import summary
#         import torchvision.models as th_models

#         self.thmodel = th_models.video.r2plus1d_18(
#             pretrained = True, 
#             progress = True 
#         )
#         print(self.thmodel)
#         summary(self.thmodel,(1,)+self.des_shape)

#     def forward(self, obs: torch.Tensor) -> torch.Tensor:
#         obs = self.modify_obs(obs)       
#         return self.thmodel(obs)


#     def modify_obs(self, obs: torch.Tensor) -> torch.Tensor:
#         """
#         All pre-trained models expect input images normalized in the 
#         same way, i.e. mini-batches of 3-channel RGB videos of shape 
#         (3 x T x H x W), where H and W are expected to be 112, and T 
#         is a number of video frames in a clip. The images have to be 
#         loaded in to a range of [0, 1].

#         Args:
#             obs (torch.Tensor): _description_

#         Returns:
#             torch.Tensor: _description_
#         """


#         if torch.any(obs > 1.0):
#             obs = obs / 255.0
#             print("NORMALIZED IMAGES")
#         else:
#             print("NO NO norm NO NO")


#         print("Before++++++ ", obs.shape)
#         plotter3(obs, rgb_gray=3, name="pytorch before")
#         obs = torch.reshape(obs, (obs.shape[0],)+self.des_shape)
#         print("After++++++ ", obs.shape)
#         plotter3(obs, rgb_gray=3, name="pytorch AFTER")

#         assert(False)

#         # mod_obs = torch.swapaxes(obs, 1, 2)

#         assert(torch.all(mod_obs >= 0) and torch.all(mod_obs <= 1.0 ))

#         return mod_obs


def plotter3(observation:torch.Tensor, rgb_gray=3, name: str = "PLotter"):
    """Plot images

    Args:
        obs (np.ndarray): Image in 3d, 4d, or 5d format.
            3d image format =(rgb*n_stack, height, width)
            4d image format =(batch, rgb*n_stack, height, width)
            5d image format =(batch, rgb, n_stack, height, width)
        rgb_gray (int, optional): 3 for rgb and 1 for grayscale. Defaults to 1.
    """

    import matplotlib.pyplot as plt

    dim = len(observation.shape)
    assert(dim in [3,4,5])

    print("TYPE OF OBSERVATION OBJECT=",type(observation))
    print("OBSERVATION SHAPE INSIDE PLOTTER=",observation.shape)

    if torch.is_tensor(observation):
        print("CONVERTED TENSOR TO NUMPY")
        obs = observation.detach().cpu().numpy()
    elif isinstance(observation, np.ndarray):
        print("OBJECT IS ALREADY IS NUMPY FORMAT")
        obs = observation.copy()
    else:
        raise Exception("INCORRECT FORMAT")

    if dim==3:
        rows = 1
        columns = obs.shape[0] // rgb_gray
        print("3-rows",rows, "3-columns",columns)
    elif dim==4:
        rows = obs.shape[0]
        columns = obs.shape[1] // rgb_gray
        print("4-rows",rows, "4-columns",columns)
    else: # dim==5
        rows = obs.shape[0]
        columns = obs.shape[2]
        print("5-rows",rows, "5-columns",columns)


    fig, axs = plt.subplots(nrows=rows, ncols=columns, squeeze=False)
    fig.suptitle("Observation PLOTTER3D")

    for row in range(0, rows):
        for col in range(0, columns):
            if dim==3:
                print("DIM3")
                img = obs[col * rgb_gray : col * rgb_gray + rgb_gray, :, :]
            elif dim==4:
                print("DIM4")
                img = obs[row, col * rgb_gray : col * rgb_gray + rgb_gray, :, :]
            elif dim==5:
                print("DIM5")
                img = obs[row, :, col, :, :]
            else:
                raise Exception("NO SUCH DIMENSION")

            print("IMG SHAPE BEFORE PLOTTER3D", img.shape)
            img = img.transpose(1, 2, 0)
            print("IMG SHAPE AFTER TRANSPOSE IN PLOTTER3D", img.shape)
            axs[row, col].imshow(img)
            axs[row, col].set_title(f"{name}")
       
    plt.show()
    plt.pause(3)
    plt.close()


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
    kwargs["policy_kwargs"] = dict(
        # activation_fn=th.nn.Tanh, # default activation used
        net_arch=[],
    )
    return kwargs


def customnaturecnn(config):
    kwargs = {}
    kwargs["policy_kwargs"] = dict(
        # features_extractor_class=NatureCNN,
        # activation_fn=th.nn.Tanh, # default activation used
        net_arch=[128, dict(pi=[32, 32], vf=[32, 32])],
    )
    return kwargs


def dreamer(config):
    kwargs = {}
    kwargs["policy_kwargs"] = dict(
        features_extractor_class=Dreamer,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=[],
    )
    return kwargs


def l5kit(config):
    kwargs = {}
    kwargs["policy_kwargs"] = dict(
        features_extractor_class=L5Kit,
        features_extractor_kwargs=dict(features_dim=128),
        normalize_images=False,
        net_arch=[],
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
    kwargs["seed"] = 42
    kwargs["batch_size"] = 64

    return kwargs

def r2plus1d_18(config):
    kwargs = {}
    kwargs["policy_kwargs"] = dict(
        features_extractor_class=R2plus1D_18,
        features_extractor_kwargs=dict(features_dim=400, config=config),
        net_arch=[],
    )

    return kwargs