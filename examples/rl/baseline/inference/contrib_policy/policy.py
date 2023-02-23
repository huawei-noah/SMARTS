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
from contrib_policy.filter_obs import FilterObs
from contrib_policy.frame_stack import FrameStack
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

        self._model = model
        self._config = config
        self._filter_obs = FilterObs(top_down_rgb=top_down_rgb)
        self._frame_stack = FrameStack(
            input_space=self._filter_obs.observation_space,
            num_stack=config.num_stack,
            stack_axis=0
        )
        self._config.observation_space = self._frame_stack.observation_space
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
        # if obs["steps_completed"] == 1:
        #     self.reset()

        processed_obs = self.process(obs)
        tensor_obs = torch.Tensor(processed_obs).to(self._config.device)
        hidden = self._model.network(tensor_obs / 255.0)
        logits = self._model.actor(hidden)
        probs = Categorical(logits=logits)
        action = probs.mode() 
        return action.cpu().numpy()

        # hide mission
        # assign random route mission
        # Tell the agent who is the leader

    def process(self, obs):
        obs = self._filter_obs.filter(obs)
        obs = self._frame_stack.step(obs)
        return obs

    def store(self,name,step,value):
        assert name in ["obs","actions","logprobs","rewards","dones","values"]
        assert type(step) == int and step >= 0
        attr = getattr(self, name)
        attr[step] = value

    def reset(self):
        # fmt: off
        self._frame_stack.reset()
        # Storage setup
        self.obs = torch.zeros((self._config.num_steps, self._config.num_envs) + self._config.observation_space.shape).to(self._config.device)
        self.actions = torch.zeros((self._config.num_steps, self._config.num_envs) + self._config.action_space.shape).to(self._config.device)
        self.logprobs = torch.zeros((self._config.num_steps, self._config.num_envs)).to(self._config.device)
        self.rewards = torch.zeros((self._config.num_steps, self._config.num_envs)).to(self._config.device)
        self.dones = torch.zeros((self._config.num_steps, self._config.num_envs)).to(self._config.device)
        self.values = torch.zeros((self._config.num_steps, self._config.num_envs)).to(self._config.device)
        # fmt: on




class Model(nn.Module):
    def __init__(self, in_channels, out_actions):
        super(Model, self).__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(in_channels, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, out_actions), std=0.01)
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


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer
