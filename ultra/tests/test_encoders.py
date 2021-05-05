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
import inspect
import os
import unittest

import numpy as np
import torch

from smarts.core.events import Events
from ultra.baselines.common.social_vehicle_config import get_social_vehicle_configs
from ultra.baselines.common.yaml_loader import load_yaml
from ultra.baselines.dqn.dqn.network import DQNWithSocialEncoder
from ultra.baselines.dqn.dqn.policy import DQNPolicy


class EncoderTest(unittest.TestCase):
    # def test_pointnet_encoder(self):
    #     X, y = self._generate_random_dataset(batch_size=64, feature_size=4)

    def test_dqn_learning(self):
        policy_class = DQNPolicy
        policy_class_module_file = inspect.getfile(policy_class)
        policy_class_module_directory = os.path.dirname(policy_class_module_file)
        policy_params = load_yaml(
            os.path.join(policy_class_module_directory, "params.yaml")
        )

        agent = policy_class(policy_params=policy_params)

        state = self._generate_random_state(
            low_dim_state_size=(47,), social_vehicles_size=(5, 4)
        )
        for _ in range(100):
            action = agent.act(state, explore=False)
            next_state = self._generate_random_state(
                low_dim_state_size=(47,), social_vehicles_size=(5, 4)
            )
            events = Events(
                collisions=False,
                off_road=False,
                reached_goal=False,
                reached_max_episode_steps=False,
                off_route=False,
                # on_shoulder=False,
                wrong_way=False,
                not_moving=False,
            )
            agent.step(
                state, action, 0.0, next_state, False, {"logs": {"events": events}}
            )
            state = next_state

        for step in range(1001):
            loss = agent.learn()
            if step % 10 == 0:
                print(f"Loss at step {step}:", loss["loss/all"]["data"].item())

        # Learn from the replay buffer states.

    def test_dqn_pointnet_encoder(self):
        NUM_LOW_DIM_STATES = 20
        NUM_SOCIAL_FEATURES = 4
        NUM_SOCIAL_VEHICLES = 5
        LEARNING_RATE = 1e-5

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", device)

        # state_size = NUM_LOW_DIM_STATES + NUM_SOCIAL_FEATURES * NUM_SOCIAL_VEHICLES
        state_size = 148
        action_size = 1
        num_actions = [action_size]
        social_vehicle_config_dict = {
            "encoder_key": "pointnet_encoder",
            "social_policy_hidden_units": 128,
            "social_policy_init_std": 0.5,
            "social_capacity": NUM_SOCIAL_VEHICLES,
            "num_social_features": NUM_SOCIAL_FEATURES,
            "seed": 2,
        }
        social_vehicle_config = get_social_vehicle_configs(**social_vehicle_config_dict)

        network_params = {
            "num_actions": num_actions,
            "state_size": state_size,
            "social_feature_encoder_class": social_vehicle_config["encoder"]["social_feature_encoder_class"],
            "social_feature_encoder_params": social_vehicle_config["encoder"]["social_feature_encoder_params"],
        }
        network = DQNWithSocialEncoder(**network_params).to(device)

        for parameter in network.social_feature_encoder.parameters():
            print("inital parameter:", parameter)
            break

        optimizer = torch.optim.Adam(params=network.parameters(), lr=LEARNING_RATE)
        loss_function = torch.nn.MSELoss(reduction="none")

        batch_size = 2

        X, y = self._generate_random_dataset(
            batch_size,
            NUM_LOW_DIM_STATES + NUM_SOCIAL_FEATURES * NUM_SOCIAL_VEHICLES,
            action_size
        )

        low_dim_states = torch.from_numpy(X[:, :NUM_LOW_DIM_STATES]).to(device)
        social_vehicles = [
            torch.from_numpy(social_vehicles_features).to(device)
            for social_vehicles_features in np.resize(
                X[:, NUM_LOW_DIM_STATES:],
                (batch_size, NUM_SOCIAL_VEHICLES, NUM_SOCIAL_FEATURES)
            )
        ]
        actions = torch.from_numpy(y).to(device)

        batch = {
            "low_dim_states": low_dim_states,
            "social_vehicles": social_vehicles,
        }

        for step in range(1000):
            output = network(batch)[0]
            loss = loss_function(output, actions).mean()
            if step % 100 == 0:
                print(f"Loss at step {step}:", loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        for parameter in network.social_feature_encoder.parameters():
            print("final parameter:", parameter)
            break

    def _generate_random_dataset(self, batch_size, input_size, output_size):
        return (
            np.random.uniform(low=-1.0, high=1.0, size=(batch_size, input_size)).astype(np.float32),
            np.random.uniform(low=-1.0, high=1.0, size=(batch_size, output_size)).astype(np.float32)
        )

    def _generate_random_state(self, low_dim_state_size, social_vehicles_size):
        state = {
            "low_dim_states": np.random.uniform(
                low=-1.0, high=1.0, size=low_dim_state_size
            ).astype(np.float32),
            "social_vehicles": np.random.uniform(
                low=-1.0, high=1.0, size=social_vehicles_size
            ).astype(np.float32),
        }
        return state


if __name__ == "__main__":
    encoder_test = EncoderTest()
    encoder_test.test_dqn_learning()
