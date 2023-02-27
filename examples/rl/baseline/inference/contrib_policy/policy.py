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
from contrib_policy.format_action import FormatAction
from torch.distributions.categorical import Categorical

from smarts.core.agent import Agent



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
        self.format_action = FormatAction()

        # fmt: off
        self._frame_stack.reset()
        # Storage setup
        self._obs = torch.zeros((self._config.num_steps, self._config.num_envs) + self._config.observation_space.shape).to(self._config.device)
        self._actions = torch.zeros((self._config.num_steps, self._config.num_envs) + self._config.action_space.shape).to(self._config.device)
        self._logprobs = torch.zeros((self._config.num_steps, self._config.num_envs)).to(self._config.device)
        self._rewards = torch.zeros((self._config.num_steps, self._config.num_envs)).to(self._config.device)
        self._dones = torch.zeros((self._config.num_steps, self._config.num_envs)).to(self._config.device)
        self._values = torch.zeros((self._config.num_steps, self._config.num_envs)).to(self._config.device)
        # fmt: on

        self.global_step = -1
        self.next_rollout()

        print("Policy initialised.")

    def act(self, obs):
        """Act function to be implemented by user.

        Args:
            obs (Any): A dictionary of observation for each ego agent step.

        Returns:
            Dict[str, Any]: A dictionary of actions for each ego agent.
        """
        processed_obs = self._process(obs)
        hidden = self._model.network(processed_obs / 255.0)
        logits = self._model.actor(hidden)
        probs = Categorical(logits=logits)
        action_mode = probs.mode() 
        formatted_action = self.format_action.format(action_mode.cpu().numpy())
        return formatted_action

        # hide mission
        # assign random route mission
        # Tell the agent who is the leader

    def _process(self, obs):
        if obs["steps_completed"] == 1:
            # Reset memory because episode was reset.
            self._frame_stack.reset()
        obs = self._filter_obs.filter(obs)
        obs = self._frame_stack.stack(obs)
        obs = torch.Tensor(np.expand_dims(obs,0)).to(self._config.device)
        return obs

    def next_rollout(self):
        self.step=-1

    def increment_step(self):
        self.step += 1
        self.global_step += 1

    @property
    def obs(self):
        return self._obs

    @obs.setter
    def obs(self, x):  
        self._obs[self.step] = self._process(x)

    @property
    def dones(self):
        return self._dones

    @dones.setter
    def dones(self, x:bool, obs):  
        if obs["steps_completed"] == 1:
            if self.global_step == 0:
                assert x == False
            else:
                assert x == True
        else:
            assert x == False
        self._dones[self.step] = torch.Tensor([int(x)]).to(self._config.device) 

    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, x):
        self._values[self.step]= x

    @property
    def actions(self):
        return self._actions
    
    @actions.setter
    def actions(self, x):
        self._actions[self.step]= x

    @property
    def logprobs(self):
        return self._logprobs
    
    @logprobs.setter
    def logprobs(self, x):
        self._logprobs[self.step]= x

    @property
    def rewards(self):
        return self._rewards
    
    @rewards.setter
    def rewards(self, x):
        tensor = torch.tensor(x).to(self._config.device).view(-1)
        self._rewards[self.step]= tensor 
 

class Model(nn.Module):
    def __init__(self, in_channels, out_actions):
        super(Model, self).__init__()
        # self._cnn = nn.Sequential(
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(in_channels, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(9216, 512)),
            nn.ReLU(),
        )

        # # Compute shape by doing one forward pass
        # temp_space = gym.spaces.Box(
        #     low=0,
        #     high=255,
        #     shape=(9,128,128),
        #     dtype=np.uint8,
        # )
        # with torch.no_grad():
        #     n_flatten = self._cnn(torch.as_tensor(temp_space.sample()[None]).float()).shape[1]

        # self._linear = nn.Sequential(layer_init(nn.Linear(n_flatten, 512)), nn.ReLU())
        # self.network = nn.Sequential(self._cnn, self._linear)

        self.actor = layer_init(nn.Linear(512, out_actions), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(self.network(obs))

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, action=None):

        # from contrib_policy.util import plotter3d
        # print("-----------------------------")
        # print(type(x), x.shape)
        # c = x.cpu().numpy()
        # plotter3d(obs=c,rgb_gray=3,channel_order="first",pause=0)
        # print("-----------------------------")

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


def optimize(model, config, agent):
    # bootstrap value if not done
    with torch.no_grad():
        next_value = model.get_value(next_obs).reshape(1, -1)
        if config.gae:
            advantages = torch.zeros_like(agent.rewards).to(config.device)
            lastgaelam = 0
            for t in reversed(range(config.num_steps)):
                if t == config.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + config.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + config.gamma * config.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values
        else:
            returns = torch.zeros_like(rewards).to(config.device)
            for t in reversed(range(config.num_steps)):
                if t == config.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    next_return = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    next_return = returns[t + 1]
                returns[t] = rewards[t] + config.gamma * nextnonterminal * next_return
            advantages = returns - values

    # flatten the batch
    b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
    b_logprobs = logprobs.reshape(-1)
    b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)
    b_values = values.reshape(-1)

    # Optimizing the policy and value network
    b_inds = np.arange(config.batch_size)
    clipfracs = []
    for epoch in range(config.update_epochs):
        np.random.shuffle(b_inds)
        for start in range(0, config.batch_size, config.minibatch_size):
            end = start + config.minibatch_size
            mb_inds = b_inds[start:end]

            _, newlogprob, entropy, newvalue = model.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
            logratio = newlogprob - b_logprobs[mb_inds]
            ratio = logratio.exp()

            with torch.no_grad():
                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs += [((ratio - 1.0).abs() > config.clip_coef).float().mean().item()]

            mb_advantages = b_advantages[mb_inds]
            if config.norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            # Policy loss
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - config.clip_coef, 1 + config.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value loss
            newvalue = newvalue.view(-1)
            if config.clip_vloss:
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + torch.clamp(
                    newvalue - b_values[mb_inds],
                    -config.clip_coef,
                    config.clip_coef,
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

            entropy_loss = entropy.mean()
            loss = pg_loss - config.ent_coef * entropy_loss + v_loss * config.vf_coef

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

        if config.target_kl is not None:
            if approx_kl > config.target_kl:
                break