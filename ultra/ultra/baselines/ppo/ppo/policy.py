# MIT License
#
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# some parts of this implementation is inspired by https://github.com/Khrylx/PyTorch-RL
import os
import pathlib

import numpy as np
import torch
import yaml

from smarts.core.agent import Agent
from ultra.baselines.common.social_vehicle_config import get_social_vehicle_configs
from ultra.baselines.common.state_preprocessor import *
from ultra.baselines.common.yaml_loader import load_yaml
from ultra.baselines.ppo.ppo.network import PPONetwork
from ultra.utils.common import (
    compute_sum_aux_losses,
    normalize_im,
    to_2d_action,
    to_3d_action,
)


class PPOPolicy(Agent):
    def __init__(
        self,
        policy_params=None,
        checkpoint_dir=None,
    ):
        self.policy_params = policy_params
        self.batch_size = int(policy_params["batch_size"])
        self.hidden_units = int(policy_params["hidden_units"])
        self.mini_batch_size = int(policy_params["mini_batch_size"])
        self.epoch_count = int(policy_params["epoch_count"])
        self.gamma = float(policy_params["gamma"])
        self.l = float(policy_params["l"])
        self.eps = float(policy_params["eps"])
        self.actor_tau = float(policy_params["actor_tau"])
        self.critic_tau = float(policy_params["critic_tau"])
        self.entropy_tau = float(policy_params["entropy_tau"])
        self.logging_freq = int(policy_params["logging_freq"])
        self.current_iteration = 0
        self.current_log_prob = None
        self.current_value = None
        self.seed = int(policy_params["seed"])
        self.lr = float(policy_params["lr"])
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.actions = []
        self.states = []
        self.terminals = []
        self.action_size = int(policy_params["action_size"])
        self.prev_action = np.zeros(self.action_size)

        # state preprocessing
        self.social_policy_hidden_units = int(
            policy_params["social_vehicles"].get("social_policy_hidden_units", 0)
        )
        self.social_capacity = int(
            policy_params["social_vehicles"].get("social_capacity", 0)
        )
        self.observation_num_lookahead = int(
            policy_params.get("observation_num_lookahead", 0)
        )
        self.social_polciy_init_std = int(
            policy_params["social_vehicles"].get("social_polciy_init_std", 0)
        )
        self.num_social_features = int(
            policy_params["social_vehicles"].get("num_social_features", 0)
        )
        self.social_vehicle_config = get_social_vehicle_configs(
            **policy_params["social_vehicles"]
        )

        self.social_vehicle_encoder = self.social_vehicle_config["encoder"]
        self.state_description = get_state_description(
            policy_params["social_vehicles"],
            policy_params["observation_num_lookahead"],
            self.action_size,
        )
        self.state_preprocessor = StatePreprocessor(
            preprocess_state, to_2d_action, self.state_description
        )
        self.social_feature_encoder_class = self.social_vehicle_encoder[
            "social_feature_encoder_class"
        ]
        self.social_feature_encoder_params = self.social_vehicle_encoder[
            "social_feature_encoder_params"
        ]

        # others
        self.checkpoint_dir = checkpoint_dir
        self.device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device_name)
        self.save_codes = (
            policy_params["save_codes"] if "save_codes" in policy_params else None
        )

        # PPO
        self.ppo_net = PPONetwork(
            self.action_size,
            self.state_size,
            hidden_units=self.hidden_units,
            init_std=self.social_polciy_init_std,
            seed=self.seed,
            social_feature_encoder_class=self.social_feature_encoder_class,
            social_feature_encoder_params=self.social_feature_encoder_params,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.ppo_net.parameters(), lr=self.lr)
        self.step_count = 0
        if self.checkpoint_dir:
            self.load(self.checkpoint_dir)

    @property
    def state_size(self):
        # Adjusting state_size based on number of features (ego+social)
        size = sum(self.state_description["low_dim_states"].values())
        if self.social_feature_encoder_class:
            size += self.social_feature_encoder_class(
                **self.social_feature_encoder_params
            ).output_dim
        else:
            size += self.social_capacity * self.num_social_features
        return size

    def act(self, state, explore=True):
        state = self.state_preprocessor(
            state=state,
            normalize=True,
            unsqueeze=True,
            device=self.device,
            social_capacity=self.social_capacity,
            observation_num_lookahead=self.observation_num_lookahead,
            social_vehicle_config=self.social_vehicle_config,
            prev_action=self.prev_action,
        )
        with torch.no_grad():
            dist, value = self.ppo_net(state)
        if explore:  # training mode
            action = dist.sample()
            log_prob = dist.log_prob(action)

            self.current_log_prob = log_prob
            self.current_value = value

            action = torch.squeeze(action)
            action = action.data.cpu().numpy()
        else:  # testing mode
            mean = torch.squeeze(dist.loc)
            action = mean.data.cpu().numpy()
        self.step_count += 1
        return to_3d_action(action)

    def step(self, state, action, reward, next_state, done):
        # dont treat timeout as done equal to True
        max_steps_reached = state["events"].reached_max_episode_steps
        if max_steps_reached:
            done = False
        action = to_2d_action(action)

        state = self.state_preprocessor(
            state=state,
            normalize=True,
            device=self.device,
            social_capacity=self.social_capacity,
            observation_num_lookahead=self.observation_num_lookahead,
            social_vehicle_config=self.social_vehicle_config,
            prev_action=self.prev_action,
        )

        # pass social_vehicle_rep through the network
        self.log_probs.append(self.current_log_prob.to(self.device))
        self.values.append(self.current_value.to(self.device))
        self.states.append(state)
        self.rewards.append(torch.FloatTensor([reward]).to(self.device))
        self.actions.append(
            torch.FloatTensor(
                action.reshape(
                    self.action_size,
                )
            ).to(self.device)
        )
        self.terminals.append(1.0 - float(done * 1))

        output = {}
        # batch updates over multiple episodes
        if len(self.terminals) >= self.batch_size:
            output = self.learn()

        self.prev_action = action if not done else np.zeros(self.action_size)
        return output

    def compute_returns(self, rewards, masks, values, gamma=0.99, lamda=0.95):
        """This computes the lambda return using values, rewards, and indication for terminal states.
        Source of iterative form (The Generalized Advantage Estimator):
        https://danieltakeshi.github.io/2017/04/02/notes-on-the-generalized-advantage-estimation-paper/
        """
        values = values + [0]
        A_GAE = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = (
                rewards[step] - values[step] + gamma * values[step + 1] * masks[step]
            )
            A_GAE = (
                delta + gamma * lamda * A_GAE * masks[step]
            )  #  (γλ)^l * delta_{t+1} = Advantage
            # but we need to return the returns G_t{λ}, so we add to that V_t before returning
            returns.insert(0, A_GAE + values[step])
        return returns

    def make_state_from_dict(self, states, device):
        # TODO: temporary function here. this is copied from replay_buffer.py
        #  better way is to make PPO use the replay_buffer interface
        #  but may not be important for now. just make it work
        image_keys = states[0]["images"].keys()
        images = {}
        for k in image_keys:
            _images = (
                torch.cat([e[k].unsqueeze(0) for e in states], dim=0).float().to(device)
            )
            _images = normalize_im(_images)
            images[k] = _images
        low_dim_states = (
            torch.cat([e["low_dim_states"].unsqueeze(0) for e in states], dim=0)
            .float()
            .to(device)
        )
        social_vehicles = [e["social_vehicles"].float().to(device) for e in states]
        out = {
            "images": images,
            "low_dim_states": low_dim_states,
            "social_vehicles": social_vehicles,
        }
        return out

    def get_minibatch(
        self, mini_batch_size, states, actions, log_probs, returns, advantage
    ):
        """Generator that can give the next batch in the dataset.
        This returns a minibatch of bunch of tensors
        """
        batch_size = len(states)
        ids = np.random.permutation(batch_size)
        whole_mini_batchs = batch_size // mini_batch_size * mini_batch_size
        no_mini_batchs = batch_size // mini_batch_size

        # split the dataset into number of minibatchs and discard the rest
        # (ex. if you have mini_batch=32 and batch_size = 100 then the utilized portion is only 96=3*32)
        splits = np.split(ids[:whole_mini_batchs], no_mini_batchs)
        # using a generator to return different mini-batch each time.
        for i in range(len(splits)):
            states_mini_batch = [states[e] for e in splits[i]]
            states_mini_batch = self.make_state_from_dict(
                states_mini_batch, device=self.device
            )
            yield (
                states_mini_batch,
                actions[splits[i], :].to(self.device),
                log_probs[splits[i], :].to(self.device),
                returns[splits[i], :].to(self.device),
                advantage[splits[i], :].to(self.device),
            )

    def get_ratio(self, new_pi_log_probs, old_pi_log_probs):
        return (new_pi_log_probs - old_pi_log_probs).exp()

    def update(
        self, n_epochs, mini_batch_size, states, actions, log_probs, returns, advantages
    ):
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy_loss = 0
        # multiple epochs
        for _ in range(n_epochs):
            # minibatch updates
            for (
                state,
                action,
                old_pi_log_probs,
                return_batch,
                advantage,
            ) in self.get_minibatch(
                mini_batch_size, states, actions, log_probs, returns, advantages
            ):
                (dist, value), aux_losses = self.ppo_net(state, training=True)
                entropy = dist.entropy().mean()  # L_S
                new_pi_log_probs = dist.log_prob(action)

                ratio = self.get_ratio(new_pi_log_probs, old_pi_log_probs)
                L_CPI = ratio * advantage
                clipped_version = (
                    torch.clamp(ratio, 1.0 - self.eps, 1.0 + self.eps) * advantage
                )

                # loss and clipping
                actor_loss = -torch.min(L_CPI, clipped_version).mean()  # L_CLIP
                critic_loss = (
                    (return_batch - value).pow(2).mean()
                )  # L_VF (squared error loss)

                aux_losses = compute_sum_aux_losses(aux_losses)

                # overall loss
                loss = (
                    self.critic_tau * critic_loss
                    + self.actor_tau * actor_loss
                    - self.entropy_tau * entropy
                    + aux_losses
                )

                # calculate gradients and update the weights
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy_loss += entropy.item()

        average_actor_loss = total_actor_loss / (
            n_epochs * (self.batch_size / self.mini_batch_size)
        )
        average_critic_loss = total_critic_loss / (
            n_epochs * (self.batch_size / self.mini_batch_size)
        )
        average_entropy_loss = total_entropy_loss / (
            n_epochs * (self.batch_size / self.mini_batch_size)
        )

        output = {
            "loss/critic": {
                "type": "scalar",
                "data": average_critic_loss,
                "freq": self.logging_freq,
            },
            "loss/actor": {
                "type": "scalar",
                "data": average_actor_loss,
                "freq": self.logging_freq,
            },
            "loss/entropy": {
                "type": "scalar",
                "data": average_entropy_loss,
                "freq": self.logging_freq,
            },
        }

        return output

    def learn(self):

        # compute lambda returns from (rewards, values) trajectories:
        returns = self.compute_returns(
            self.rewards, self.terminals, self.values, self.gamma, self.l
        )

        # convert lists into Pytorch tensors
        states = self.states
        actions = torch.stack(self.actions)
        log_probs = torch.cat(self.log_probs).detach()
        values = torch.cat(self.values).detach()
        returns = torch.cat(returns).detach()
        advantages = returns - values

        # normalize advatages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        # PPO update for # of epochs and over # of minibatchs
        output = self.update(
            self.epoch_count,
            self.mini_batch_size,
            states,
            actions,
            log_probs,
            returns,
            advantages,
        )

        # remove previous experiences, in preparing for the next iteration
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.actions = []
        self.states = []
        self.terminals = []

        # increase the current number of iterations
        self.current_iteration += 1
        return output

    def save(self, model_dir):
        model_dir = pathlib.Path(model_dir)
        torch.save(self.ppo_net.state_dict(), model_dir / "ppo_network.pth")

    def load(self, model_dir):
        print("model loaded:", model_dir)
        model_dir = pathlib.Path(model_dir)
        map_location = None
        if self.device and self.device.type == "cpu":
            map_location = "cpu"
        self.ppo_net.load_state_dict(
            torch.load(model_dir / "ppo_network.pth", map_location=map_location)
        )

    def reset(self):
        pass
