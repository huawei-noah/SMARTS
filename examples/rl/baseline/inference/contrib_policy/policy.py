import sys
from pathlib import Path

# To import contrib_policy folder
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from typing import Any, Callable, Dict, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from contrib_policy.observation import FilterObs
from torch.distributions.categorical import Categorical

from smarts.core.agent import Agent

# def submitted_wrappers():
#     """Return environment wrappers for wrapping the evaluation environment.
#     Each wrapper is of the form: Callable[[env], env]. Use of wrappers is
#     optional. If wrappers are not used, return empty list [].

#     Returns:
#         List[wrappers]: List of wrappers. Default is empty list [].
#     """

#     from action import Action as DiscreteAction
#     from observation import Concatenate, FilterObs, SaveObs

#     from smarts.core.controllers import ActionSpaceType
#     from smarts.env.wrappers.format_action import FormatAction
#     from smarts.env.wrappers.format_obs import FormatObs
#     from smarts.env.wrappers.frame_stack import FrameStack

#     # fmt: off
#     wrappers = [
#         FormatObs,
#         lambda env: FormatAction(env=env, space=ActionSpaceType["TargetPose"]),
#         SaveObs,
#         DiscreteAction,
#         FilterObs,
#         lambda env: FrameStack(env=env, num_stack=3),
#         lambda env: Concatenate(env=env, channels_order="first"),
#     ]
#     # fmt: on

#     return wrappers


class Policy(Agent):
    """Policy class to be submitted by the user. This class will be loaded
    and tested during evaluation."""

    def __init__(self, config, model, top_down_rgb):
        """All policy initialization matters, including loading of model, is
        performed here. To be implemented by the user.
        """

        # model_path = Path(__file__).resolve().parents[0] / "saved_model.zip"
        # self.model = sb3lib.PPO.load(model_path)

        self.model = model
        self.config = config
        self.filter_obs = FilterObs(top_down_rgb)
        self.config.observation_space = self.filter_obs.observation_space
        self.reset()
        print("Policy initialised.")

    def act(self, obs):
        """Act function to be implemented by user.

        Args:
            obs (Any): A dictionary of observation for each ego agent step.

        Returns:
            Dict[str, Any]: A dictionary of actions for each ego agent.
        """
        # Reset memory because episode was reset.
        if obs["steps_completed"] == 1:
            self.reset()

        filtered_obs = self.filter_obs.filter(obs)

        # action, _ = self.model.predict(observation=obs, deterministic=True)
        # processed_act = action

        # wrapped_act = action_wrapper._discrete(action, self.saved_obs)

        return [0.1, 0.1, 0.1]  # processed_act

    def reset(self):
        # fmt: off
        # Storage setup
        self.obs = torch.zeros((self.config.num_steps, self.config.num_envs) + self.config.observation_space.shape).to(self.config.device)
        self.actions = torch.zeros((self.config.num_steps, self.config.num_envs) + self.config.action_space.shape).to(self.config.device)
        self.logprobs = torch.zeros((self.config.num_steps, self.config.num_envs)).to(self.config.device)
        self.rewards = torch.zeros((self.config.num_steps, self.config.num_envs)).to(self.config.device)
        self.dones = torch.zeros((self.config.num_steps, self.config.num_envs)).to(self.config.device)
        self.values = torch.zeros((self.config.num_steps, self.config.num_envs)).to(self.config.device)
        # fmt: on


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Model(nn.Module):
    def __init__(self, output_dim):
        super(Model, self).__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, output_dim), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)
