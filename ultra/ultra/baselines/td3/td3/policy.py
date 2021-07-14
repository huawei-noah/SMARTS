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
import pathlib, os, copy
import torch
import torch.nn.functional as F
import random
import numpy as np
import torch.optim as optim

from ultra.baselines.td3.td3.cnn_models import ActorNetwork as CNNActorNetwork
from ultra.baselines.td3.td3.cnn_models import CriticNetwork as CNNCriticNetwork
from ultra.baselines.td3.td3.fc_model import ActorNetwork as FCActorNetwork
from ultra.baselines.td3.td3.fc_model import CriticNetwork as FCCrtiicNetwork
from smarts.core.agent import Agent
from ultra.baselines.td3.td3.noise import (
    OrnsteinUhlenbeckProcess,
    LinearSchedule,
)
from ultra.utils.common import compute_sum_aux_losses, to_3d_action, to_2d_action
import ultra.adapters as adapters
from ultra.baselines.common.replay_buffer import ReplayBuffer
from ultra.baselines.common.social_vehicle_config import get_social_vehicle_configs
from ultra.baselines.common.yaml_loader import load_yaml


class TD3Policy(Agent):
    def __init__(
        self,
        policy_params=None,
        checkpoint_dir=None,
    ):
        self.policy_params = policy_params
        self.action_size = 2
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
        self.action_type = adapters.type_from_string(policy_params["action_type"])
        self.observation_type = adapters.type_from_string(
            policy_params["observation_type"]
        )
        self.reward_type = adapters.type_from_string(policy_params["reward_type"])

        if self.action_type != adapters.AdapterType.DefaultActionContinuous:
            raise Exception(
                f"TD3 baseline only supports the "
                f"{adapters.AdapterType.DefaultActionContinuous} action type."
            )

        if self.observation_type == adapters.AdapterType.DefaultObservationVector:
            observation_space = adapters.space_from_type(self.observation_type)
            low_dim_states_size = observation_space["low_dim_states"].shape[0]
            social_capacity = observation_space["social_vehicles"].shape[0]
            num_social_features = observation_space["social_vehicles"].shape[1]

            # Get information to build the encoder.
            encoder_key = policy_params["social_vehicles"]["encoder_key"]
            social_policy_hidden_units = int(
                policy_params["social_vehicles"].get("social_policy_hidden_units", 0)
            )
            social_policy_init_std = int(
                policy_params["social_vehicles"].get("social_policy_init_std", 0)
            )
            social_vehicle_config = get_social_vehicle_configs(
                encoder_key=encoder_key,
                num_social_features=num_social_features,
                social_capacity=social_capacity,
                seed=self.seed,
                social_policy_hidden_units=social_policy_hidden_units,
                social_policy_init_std=social_policy_init_std,
            )
            social_vehicle_encoder = social_vehicle_config["encoder"]
            social_feature_encoder_class = social_vehicle_encoder[
                "social_feature_encoder_class"
            ]
            social_feature_encoder_params = social_vehicle_encoder[
                "social_feature_encoder_params"
            ]

            # Calculate the state size based on the number of features (ego + social).
            state_size = low_dim_states_size
            if social_feature_encoder_class:
                state_size += social_feature_encoder_class(
                    **social_feature_encoder_params
                ).output_dim
            else:
                state_size += social_capacity * num_social_features
            # Add the action size to account for the previous action.
            state_size += self.action_size

            actor_network_class = FCActorNetwork
            critic_network_class = FCCrtiicNetwork
            network_params = {
                "state_space": state_size,
                "action_space": self.action_size,
                "seed": self.seed,
                "social_feature_encoder": social_feature_encoder_class(
                    **social_feature_encoder_params
                )
                if social_feature_encoder_class
                else None,
            }
        elif self.observation_type == adapters.AdapterType.DefaultObservationImage:
            observation_space = adapters.space_from_type(self.observation_type)
            stack_size = observation_space.shape[0]
            image_shape = (observation_space.shape[1], observation_space.shape[2])

            actor_network_class = CNNActorNetwork
            critic_network_class = CNNCriticNetwork
            network_params = {
                "input_channels": stack_size,
                "input_dimension": image_shape,
                "action_size": self.action_size,
                "seed": self.seed,
            }
        else:
            raise Exception(
                f"TD3 baseline does not support the '{self.observation_type}' "
                f"observation type."
            )

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
            observation_type=self.observation_type,
            device_name=self.device_name,
        )
        self.num_actor_updates = 0
        self.current_iteration = 0
        self.step_count = 0
        self.init_networks(actor_network_class, critic_network_class, network_params)
        if checkpoint_dir:
            self.load(checkpoint_dir)

    def init_networks(self, actor_network_class, critic_network_class, network_params):
        self.noise = [
            OrnsteinUhlenbeckProcess(
                size=(1,), theta=0.01, std=LinearSchedule(0.25), mu=0.0, x0=0.0, dt=1.0
            ),  # throttle
            OrnsteinUhlenbeckProcess(
                size=(1,), theta=0.1, std=LinearSchedule(0.05), mu=0.0, x0=0.0, dt=1.0
            ),  # steering
        ]

        self.actor = actor_network_class(**network_params).to(self.device)
        self.actor_target = actor_network_class(**network_params).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)

        self.critic_1 = critic_network_class(**network_params).to(self.device)
        self.critic_1_target = critic_network_class(**network_params).to(self.device)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_1_optimizer = optim.Adam(
            self.critic_1.parameters(), lr=self.critic_lr
        )

        self.critic_2 = critic_network_class(**network_params).to(self.device)
        self.critic_2_target = critic_network_class(**network_params).to(self.device)
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        self.critic_2_optimizer = optim.Adam(
            self.critic_2.parameters(), lr=self.critic_lr
        )

    def act(self, state, explore=True):
        state = copy.deepcopy(state)
        if self.observation_type == adapters.AdapterType.DefaultObservationVector:
            # Default vector observation type.
            state["low_dim_states"] = np.float32(
                np.append(state["low_dim_states"], self.prev_action)
            )
            state["social_vehicles"] = (
                torch.from_numpy(state["social_vehicles"]).unsqueeze(0).to(self.device)
            )
            state["low_dim_states"] = (
                torch.from_numpy(state["low_dim_states"]).unsqueeze(0).to(self.device)
            )
        else:
            # Default image observation type.
            state = torch.from_numpy(state).unsqueeze(0).to(self.device)

        self.actor.eval()

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

    def step(self, state, action, reward, next_state, done, info):
        # dont treat timeout as done equal to True
        max_steps_reached = info["logs"]["events"].reached_max_episode_steps
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
