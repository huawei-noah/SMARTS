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
import torch
from torch import nn
import numpy as np
import ultra.adapters as adapters
from smarts.core.agent import Agent
from ultra.utils.common import merge_discrete_action_spaces, to_3d_action, to_2d_action
import pathlib, os, copy
from ultra.baselines.dqn.dqn.policy import DQNPolicy
from ultra.baselines.dqn.dqn.network import DQNCNN, DQNWithSocialEncoder
from ultra.baselines.dqn.dqn.explore import EpsilonExplore
from ultra.baselines.common.replay_buffer import ReplayBuffer
from ultra.baselines.common.social_vehicle_config import get_social_vehicle_configs
from ultra.baselines.common.yaml_loader import load_yaml


class BehavioralDQNPolicy(DQNPolicy):
    def __init__(
        self,
        policy_params=None,
        checkpoint_dir=None,
    ):
        self.policy_params = policy_params
        self.lr = float(policy_params["lr"])
        self.seed = int(policy_params["seed"])
        self.train_step = int(policy_params["train_step"])
        self.target_update = float(policy_params["target_update"])
        self.warmup = int(policy_params["warmup"])
        self.gamma = float(policy_params["gamma"])
        self.batch_size = int(policy_params["batch_size"])
        self.use_ddqn = policy_params["use_ddqn"]
        self.sticky_actions = int(policy_params["sticky_actions"])
        self.epsilon_obj = EpsilonExplore(1.0, 0.05, 100000)
        self.step_count = 0
        self.update_count = 0
        self.num_updates = 0
        self.current_sticky = 0
        self.current_iteration = 0
        self.action_type = adapters.type_from_string(policy_params["action_type"])
        self.observation_type = adapters.type_from_string(
            policy_params["observation_type"]
        )
        self.reward_type = adapters.type_from_string(policy_params["reward_type"])

        if self.action_type == adapters.AdapterType.DefaultActionDiscrete:
            discrete_action_spaces = [[0], [1]]
            index_to_actions = [
                discrete_action_space.tolist()
                if not isinstance(discrete_action_space, list)
                else discrete_action_space
                for discrete_action_space in discrete_action_spaces
            ]
            action_to_indexs = {
                str(discrete_action): index
                for discrete_action, index in zip(
                    index_to_actions, np.arange(len(index_to_actions)).astype(np.int)
                )
            }
            self.index2actions = [index_to_actions]
            self.action2indexs = [action_to_indexs]
            self.merge_action_spaces = -1
            self.num_actions = [len(index_to_actions)]
            self.action_size = 1
            self.to_real_action = lambda action: self.lane_actions[action[0]]
        else:
            raise Exception(
                f"BDQN baseline does not support the '{self.action_type}' action type."
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

            network_class = DQNWithSocialEncoder
            network_params = {
                "num_actions": self.num_actions,
                "state_size": state_size,
                "social_feature_encoder_class": social_feature_encoder_class,
                "social_feature_encoder_params": social_feature_encoder_params,
            }
        elif self.observation_type == adapters.AdapterType.DefaultObservationImage:
            observation_space = adapters.space_from_type(self.observation_type)
            stack_size = observation_space.shape[0]
            image_shape = (observation_space.shape[1], observation_space.shape[2])

            network_class = DQNCNN
            network_params = {
                "n_in_channels": stack_size,
                "image_dim": image_shape,
                "num_actions": self.num_actions,
            }
        else:
            raise Exception(
                f"DQN baseline does not support the '{self.observation_type}' "
                f"observation type."
            )

        self.prev_action = np.zeros(self.action_size)
        self.checkpoint_dir = checkpoint_dir
        torch.manual_seed(self.seed)
        self.device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device_name)
        self.online_q_network = network_class(**network_params).to(self.device)
        self.target_q_network = network_class(**network_params).to(self.device)
        self.update_target_network()
        self.optimizers = torch.optim.Adam(
            params=self.online_q_network.parameters(), lr=self.lr
        )
        self.loss_func = nn.MSELoss(reduction="none")
        self.replay = ReplayBuffer(
            buffer_size=int(policy_params["replay_buffer"]["buffer_size"]),
            batch_size=int(policy_params["replay_buffer"]["batch_size"]),
            observation_type=self.observation_type,
            device_name=self.device_name,
        )
        self.reset()
        if self.checkpoint_dir:
            self.load(self.checkpoint_dir)
