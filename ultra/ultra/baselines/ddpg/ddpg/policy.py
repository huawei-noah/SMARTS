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
# code iintegrated from
#  1- https://github.com/udacity/deep-reinforcement-learning
#  2- https://github.com/sfujim/TD3/blob/master/TD3.py
#
import os
import pathlib
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from smarts.core.agent import Agent
from ultra.baselines.common.replay_buffer import ReplayBuffer
from ultra.baselines.common.social_vehicle_config import get_social_vehicle_configs
from ultra.baselines.common.state_preprocessor import *
from ultra.baselines.common.yaml_loader import load_yaml
from ultra.baselines.ddpg.ddpg.fc_model import ActorNetwork, CriticNetwork
from ultra.baselines.ddpg.ddpg.noise import LinearSchedule, OrnsteinUhlenbeckProcess
from ultra.utils.common import compute_sum_aux_losses, to_2d_action, to_3d_action


class TD3Policy(Agent):
    def __init__(
        self,
        policy_params=None,
        checkpoint_dir=None,
    ):
        self.policy_params = policy_params
        self.action_size = int(policy_params["action_size"])
        self.action_range = np.asarray([[-1.0, 1.0], [-1.0, 1.0]], dtype=np.float32)
        self.actor_lr = float(policy_params["actor_lr"])
        self.critic_lr = float(policy_params["critic_lr"])
        self.critic_wd = float(policy_params["critic_wd"])
        self.actor_wd = float(policy_params["actor_wd"])
        self.noise_clip = float(policy_params["noise_clip"])
        self.policy_noise = float(policy_params["policy_noise"])
        self.update_rate = int(policy_params["update_rate"])
        self.policy_delay = int(policy_params["policy_delay"])
        self.warmup = int(policy_params["warmup"])
        self.critic_tau = float(policy_params["critic_tau"])
        self.actor_tau = float(policy_params["actor_tau"])
        self.gamma = float(policy_params["gamma"])
        self.batch_size = int(policy_params["batch_size"])
        self.sigma = float(policy_params["sigma"])
        self.theta = float(policy_params["theta"])
        self.dt = float(policy_params["dt"])
        self.action_low = torch.tensor([[each[0] for each in self.action_range]])
        self.action_high = torch.tensor([[each[1] for each in self.action_range]])
        self.seed = int(policy_params["seed"])
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
        self.memory = ReplayBuffer(
            buffer_size=int(policy_params["replay_buffer"]["buffer_size"]),
            batch_size=int(policy_params["replay_buffer"]["batch_size"]),
            state_preprocessor=self.state_preprocessor,
            device_name=self.device_name,
        )
        self.num_actor_updates = 0
        self.current_iteration = 0
        self.step_count = 0
        self.init_networks()
        if checkpoint_dir:
            self.load(checkpoint_dir)

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

    def init_networks(self):
        self.noise = [
            OrnsteinUhlenbeckProcess(
                size=(1,), theta=0.01, std=LinearSchedule(0.25), mu=0.0, x0=0.0, dt=1.0
            ),  # throttle
            OrnsteinUhlenbeckProcess(
                size=(1,), theta=0.1, std=LinearSchedule(0.05), mu=0.0, x0=0.0, dt=1.0
            ),  # steering
        ]
        self.actor = ActorNetwork(
            self.state_size,
            self.action_size,
            self.seed,
            social_feature_encoder=self.social_feature_encoder_class(
                **self.social_feature_encoder_params
            )
            if self.social_feature_encoder_class
            else None,
        ).to(self.device)
        self.actor_target = ActorNetwork(
            self.state_size,
            self.action_size,
            self.seed,
            social_feature_encoder=self.social_feature_encoder_class(
                **self.social_feature_encoder_params
            )
            if self.social_feature_encoder_class
            else None,
        ).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)

        self.critic_1 = CriticNetwork(
            self.state_size,
            self.action_size,
            self.seed,
            social_feature_encoder=self.social_feature_encoder_class(
                **self.social_feature_encoder_params
            )
            if self.social_feature_encoder_class
            else None,
        ).to(self.device)
        self.critic_1_target = CriticNetwork(
            self.state_size,
            self.action_size,
            self.seed,
            social_feature_encoder=self.social_feature_encoder_class(
                **self.social_feature_encoder_params
            )
            if self.social_feature_encoder_class
            else None,
        ).to(self.device)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_1_optimizer = optim.Adam(
            self.critic_1.parameters(), lr=self.critic_lr
        )

        self.critic_2 = CriticNetwork(
            self.state_size,
            self.action_size,
            self.seed,
            social_feature_encoder=self.social_feature_encoder_class(
                **self.social_feature_encoder_params
            )
            if self.social_feature_encoder_class
            else None,
        ).to(self.device)
        self.critic_2_target = CriticNetwork(
            self.state_size,
            self.action_size,
            self.seed,
            social_feature_encoder=self.social_feature_encoder_class(
                **self.social_feature_encoder_params
            )
            if self.social_feature_encoder_class
            else None,
        ).to(self.device)
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        self.critic_2_optimizer = optim.Adam(
            self.critic_2.parameters(), lr=self.critic_lr
        )

    def act(self, state, explore=True):
        self.actor.eval()
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
        # print(state)
        action = self.actor(state).cpu().data.numpy().flatten()

        noise = [self.noise[0].sample(), self.noise[1].sample()]
        if explore:
            action[0] += noise[0]
            action[1] += noise[1]

        self.actor.train()
        action_low, action_high = (
            self.action_low.data.cpu().numpy(),
            self.action_high.data.cpu().numpy(),
        )
        action = np.clip(action, action_low, action_high)[0]

        return to_3d_action(action)

    def step(self, state, action, reward, next_state, done):
        # dont treat timeout as done equal to True
        max_steps_reached = state["events"].reached_max_episode_steps
        reset_noise = False
        if max_steps_reached:
            done = False
            reset_noise = True

        output = {}
        action = to_2d_action(action)
        self.memory.add(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=float(done),
            social_capacity=self.social_capacity,
            observation_num_lookahead=self.observation_num_lookahead,
            social_vehicle_config=self.social_vehicle_config,
            prev_action=self.prev_action,
        )
        self.step_count += 1
        if reset_noise:
            self.reset()
        if (
            len(self.memory) > max(self.batch_size, self.warmup)
            and (self.step_count + 1) % self.update_rate == 0
        ):
            output = self.learn()

        self.prev_action = action if not done else np.zeros(self.action_size)
        return output

    def reset(self):
        self.noise[0].reset_states()
        self.noise[1].reset_states()

    def learn(self):
        output = {}
        states, actions, rewards, next_states, dones, others = self.memory.sample(
            device=self.device
        )
        # print("????")
        actions = actions.squeeze(dim=1)
        next_actions = self.actor_target(next_states)
        noise = torch.randn_like(next_actions).mul(self.policy_noise)
        noise = noise.clamp(-self.noise_clip, self.noise_clip)
        next_actions += noise
        next_actions = torch.max(
            torch.min(next_actions, self.action_high.to(self.device)),
            self.action_low.to(self.device),
        )

        target_Q1 = self.critic_1_target(next_states, next_actions)
        target_Q2 = self.critic_2_target(next_states, next_actions)
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = (rewards + ((1 - dones) * self.gamma * target_Q)).detach()

        # Optimize Critic 1:
        current_Q1, aux_losses_Q1 = self.critic_1(states, actions, training=True)
        loss_Q1 = F.mse_loss(current_Q1, target_Q) + compute_sum_aux_losses(
            aux_losses_Q1
        )
        self.critic_1_optimizer.zero_grad()
        loss_Q1.backward()
        self.critic_1_optimizer.step()

        # Optimize Critic 2:
        current_Q2, aux_losses_Q2 = self.critic_2(states, actions, training=True)
        loss_Q2 = F.mse_loss(current_Q2, target_Q) + compute_sum_aux_losses(
            aux_losses_Q2
        )
        self.critic_2_optimizer.zero_grad()
        loss_Q2.backward()
        self.critic_2_optimizer.step()

        # delayed actor updates
        if (self.step_count + 1) % self.policy_delay == 0:
            critic_out = self.critic_1(states, self.actor(states), training=True)
            actor_loss, actor_aux_losses = -critic_out[0], critic_out[1]
            actor_loss = actor_loss.mean() + compute_sum_aux_losses(actor_aux_losses)
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            self.soft_update(self.actor_target, self.actor, self.actor_tau)

            self.num_actor_updates += 1
            output = {
                "loss/critic_1": {
                    "type": "scalar",
                    "data": loss_Q1.data.cpu().numpy(),
                    "freq": 10,
                },
                "loss/actor": {
                    "type": "scalar",
                    "data": actor_loss.data.cpu().numpy(),
                    "freq": 10,
                },
            }

        self.soft_update(self.critic_1_target, self.critic_1, self.critic_tau)
        self.soft_update(self.critic_2_target, self.critic_2, self.critic_tau)
        self.current_iteration += 1
        return output

    def soft_update(self, target, src, tau):
        for target_param, param in zip(target.parameters(), src.parameters()):
            target_param.detach_()
            target_param.copy_(target_param * (1.0 - tau) + param * tau)

    def load(self, model_dir):
        model_dir = pathlib.Path(model_dir)
        map_location = None
        if self.device and self.device.type == "cpu":
            map_location = "cpu"
        self.actor.load_state_dict(
            torch.load(model_dir / "actor.pth", map_location=map_location)
        )
        self.actor_target.load_state_dict(
            torch.load(model_dir / "actor_target.pth", map_location=map_location)
        )
        self.critic_1.load_state_dict(
            torch.load(model_dir / "critic_1.pth", map_location=map_location)
        )
        self.critic_1_target.load_state_dict(
            torch.load(model_dir / "critic_1_target.pth", map_location=map_location)
        )
        self.critic_2.load_state_dict(
            torch.load(model_dir / "critic_2.pth", map_location=map_location)
        )
        self.critic_2_target.load_state_dict(
            torch.load(model_dir / "critic_2_target.pth", map_location=map_location)
        )

    def save(self, model_dir):
        model_dir = pathlib.Path(model_dir)
        torch.save(self.actor.state_dict(), model_dir / "actor.pth")
        torch.save(
            self.actor_target.state_dict(),
            model_dir / "actor_target.pth",
        )
        torch.save(self.critic_1.state_dict(), model_dir / "critic_1.pth")
        torch.save(
            self.critic_1_target.state_dict(),
            model_dir / "critic_1_target.pth",
        )
        torch.save(self.critic_2.state_dict(), model_dir / "critic_2.pth")
        torch.save(
            self.critic_2_target.state_dict(),
            model_dir / "critic_2_target.pth",
        )
