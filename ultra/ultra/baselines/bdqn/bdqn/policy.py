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
import os
import pathlib

import numpy as np
import torch
from torch import nn

from smarts.core.agent import Agent
from ultra.baselines.bdqn.bdqn.explore import EpsilonExplore
from ultra.baselines.bdqn.bdqn.network import *
from ultra.baselines.bdqn.bdqn.network import DQNWithSocialEncoder
from ultra.baselines.common.replay_buffer import ReplayBuffer
from ultra.baselines.common.social_vehicle_config import get_social_vehicle_configs
from ultra.baselines.common.state_preprocessor import *
from ultra.baselines.common.yaml_loader import load_yaml
from ultra.baselines.dqn.dqn.policy import DQNPolicy
from ultra.utils.common import merge_discrete_action_spaces, to_2d_action, to_3d_action


class BehavioralDQNPolicy(DQNPolicy):
    def __init__(
        self,
        policy_params=None,
        checkpoint_dir=None,
    ):
        self.policy_params = policy_params
        network_class = DQNWithSocialEncoder
        self.epsilon_obj = EpsilonExplore(1.0, 0.05, 100000)

        discrete_action_spaces = [[0], [1]]
        action_size = discrete_action_spaces
        self.merge_action_spaces = -1

        self.step_count = 0
        self.update_count = 0
        self.num_updates = 0
        self.current_sticky = 0
        self.current_iteration = 0

        lr = float(policy_params["lr"])
        seed = int(policy_params["seed"])
        self.train_step = int(policy_params["train_step"])
        self.target_update = float(policy_params["target_update"])
        self.device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device_name)
        self.warmup = int(policy_params["warmup"])
        self.gamma = float(policy_params["gamma"])
        self.batch_size = int(policy_params["batch_size"])
        self.use_ddqn = policy_params["use_ddqn"]
        self.sticky_actions = int(policy_params["sticky_actions"])
        prev_action_size = 1
        self.prev_action = np.zeros(prev_action_size)

        index_to_actions = [
            e.tolist() if not isinstance(e, list) else e for e in action_size
        ]

        action_to_indexs = {
            str(k): v
            for k, v in zip(
                index_to_actions, np.arange(len(index_to_actions)).astype(np.int)
            )
        }
        self.index2actions, self.action2indexs = (
            [index_to_actions],
            [action_to_indexs],
        )
        self.num_actions = [len(index_to_actions)]

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
            prev_action_size,
        )
        self.social_feature_encoder_class = self.social_vehicle_encoder[
            "social_feature_encoder_class"
        ]
        self.social_feature_encoder_params = self.social_vehicle_encoder[
            "social_feature_encoder_params"
        ]

        self.checkpoint_dir = checkpoint_dir
        self.reset()

        torch.manual_seed(seed)
        network_params = {
            "state_size": self.state_size,
            "social_feature_encoder_class": self.social_feature_encoder_class,
            "social_feature_encoder_params": self.social_feature_encoder_params,
        }
        self.online_q_network = network_class(
            num_actions=self.num_actions,
            **(network_params if network_params else {}),
        ).to(self.device)
        self.target_q_network = network_class(
            num_actions=self.num_actions,
            **(network_params if network_params else {}),
        ).to(self.device)
        self.update_target_network()

        self.optimizers = torch.optim.Adam(
            params=self.online_q_network.parameters(), lr=lr
        )
        self.loss_func = nn.MSELoss(reduction="none")

        if self.checkpoint_dir:
            self.load(self.checkpoint_dir)

        self.action_space_type = "lane"
        self.to_real_action = lambda action: self.lane_actions[action[0]]
        self.state_preprocessor = StatePreprocessor(
            preprocess_state, self.lane_action_to_index, self.state_description
        )
        self.replay = ReplayBuffer(
            buffer_size=int(policy_params["replay_buffer"]["buffer_size"]),
            batch_size=int(policy_params["replay_buffer"]["batch_size"]),
            state_preprocessor=self.state_preprocessor,
            device_name=self.device_name,
        )
