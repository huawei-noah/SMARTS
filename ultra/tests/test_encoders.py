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
from smarts.zoo.registry import make
from ultra.baselines.common.social_vehicle_config import get_social_vehicle_configs
from ultra.baselines.common.social_vehicles_encoders.pointnet_encoder import PNEncoder
from ultra.baselines.common.social_vehicles_encoders.pointnet_encoder_batched import (
    PNEncoderBatched,
)
from ultra.baselines.common.social_vehicles_encoders.precog_encoder import (
    PrecogFeatureExtractor,
)
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

    def test_dqn_precog_encoder(self):
        pass

    def test_dqn_encoder(self):
        ENCODER_NAME = "pointnet_encoder_batched"
        NUM_LOW_DIM_STATES = 20
        NUM_SOCIAL_FEATURES = 4
        NUM_SOCIAL_VEHICLES = 5
        LEARNING_RATE = 1e-5

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", device)

        social_vehicle_config_dict = {
            "encoder_key": ENCODER_NAME,
            "social_policy_hidden_units": 128,
            "social_policy_init_std": 0.5,
            "social_capacity": NUM_SOCIAL_VEHICLES,
            "num_social_features": NUM_SOCIAL_FEATURES,
            "seed": 2,
        }
        social_vehicle_config = get_social_vehicle_configs(**social_vehicle_config_dict)
        social_vehicle_config["encoder"]["social_feature_encoder_params"][
            "bias"
        ] = False
        print("social_vehicle_config:", social_vehicle_config)

        temp_encoder = social_vehicle_config["encoder"]["social_feature_encoder_class"](
            **social_vehicle_config["encoder"]["social_feature_encoder_params"]
        )
        NUM_ENCODED_STATES = temp_encoder.output_dim
        del temp_encoder

        state_size = NUM_ENCODED_STATES + NUM_LOW_DIM_STATES
        action_size = 1
        num_actions = [action_size]
        batch_size = 2

        network_params = {
            "num_actions": num_actions,
            "state_size": state_size,
            "social_feature_encoder_class": social_vehicle_config["encoder"][
                "social_feature_encoder_class"
            ],
            "social_feature_encoder_params": social_vehicle_config["encoder"][
                "social_feature_encoder_params"
            ],
        }
        network = DQNWithSocialEncoder(**network_params).to(device)

        for parameter in network.social_feature_encoder.parameters():
            print("inital parameter:", parameter)
            break

        optimizer = torch.optim.Adam(params=network.parameters(), lr=LEARNING_RATE)
        loss_function = torch.nn.MSELoss(reduction="none")

        X, y = self._generate_random_dataset(
            batch_size,
            NUM_LOW_DIM_STATES + NUM_SOCIAL_FEATURES * NUM_SOCIAL_VEHICLES,
            action_size,
        )

        low_dim_states = torch.from_numpy(X[:, :NUM_LOW_DIM_STATES]).to(device)
        social_vehicles = [
            torch.from_numpy(social_vehicles_features).to(device)
            for social_vehicles_features in np.resize(
                X[:, NUM_LOW_DIM_STATES:],
                (batch_size, NUM_SOCIAL_VEHICLES, NUM_SOCIAL_FEATURES),
            )
        ]
        actions = torch.from_numpy(y).to(device)

        batch = {
            "low_dim_states": low_dim_states,
            "social_vehicles": social_vehicles,
        }

        loss = float("inf")
        step = 0
        while loss > 0.001:
            output = network(batch)[0]
            loss_tensor = loss_function(output, actions).mean()
            optimizer.zero_grad()
            loss_tensor.backward()
            optimizer.step()

            loss = loss_tensor.item()
            if step % 100 == 0:
                print(f"Loss at step {step}:", loss)
            step += 1

        for parameter in network.social_feature_encoder.parameters():
            print("final parameter:", parameter)
            break

    def test_encoder_observations(self):
        SOCIAL_POLICY_HIDDEN_UNITS = 128
        NUM_SOCIAL_FEATURES = 4
        SOCIAL_CAPACITY = 2
        SEED = 2

        # input_array = np.zeros(
        #     shape=(SOCIAL_CAPACITY, NUM_SOCIAL_FEATURES), dtype=np.float32
        # )
        input_array = np.random.uniform(
            low=-1.0, high=1.0, size=(SOCIAL_CAPACITY, NUM_SOCIAL_FEATURES)
        ).astype(np.float32)
        input_tensor = torch.from_numpy(input_array).unsqueeze(axis=0)

        print(f"input_tensor:\n{input_tensor}")

        # Test the precog encoder.
        precog_encoder = PrecogFeatureExtractor(
            hidden_units=SOCIAL_POLICY_HIDDEN_UNITS,
            n_social_features=NUM_SOCIAL_FEATURES,
            social_capacity=SOCIAL_CAPACITY,
            embed_dim=8,
            seed=SEED,
            bias=False,
        )
        precog_encoder.eval()
        with torch.no_grad():
            precog_result = precog_encoder(input_tensor)[0][0]
        print("precog output:", precog_result)
        print("precog output shape:", precog_result.shape)
        # expected_precog_result = torch.zeros_like(precog_result)
        # self.assertTrue(torch.equal(precog_result, expected_precog_result))

        # # Test the pointnet encoder.
        # pointnet_encoder = PNEncoder(
        #     input_dim=NUM_SOCIAL_FEATURES,
        #     global_features=True,
        #     feature_transform=True,
        #     nc=8,
        #     transform_loss_weight=0.1,
        #     bias=False,
        # )
        # pointnet_encoder.eval()
        # with torch.no_grad():
        #     pointnet_result = pointnet_encoder(input_tensor)[0][0]
        # print("pointnet output:", pointnet_result)
        # print("pointnet output shape:", pointnet_result.shape)
        # expected_pointnet_result = torch.zeros_like(pointnet_result)
        # self.assertTrue(torch.equal(pointnet_result, expected_pointnet_result))

        # # Test the pointnet batched encoder.
        # pointnet_batched_encoder = PNEncoderBatched(
        #     input_dim=NUM_SOCIAL_FEATURES,
        #     global_features=True,
        #     feature_transform=True,
        #     nc=8,
        #     transform_loss_weight=0.1,
        #     bias=False,
        # )
        # pointnet_batched_encoder.eval()
        # with torch.no_grad():
        #     pointnet_batched_result = pointnet_batched_encoder(input_tensor)[0][0]
        # print("pointnet batched output", pointnet_batched_result)
        # print("pointnet batched output shape:", pointnet_batched_result.shape)
        # expected_pointnet_batched_result = torch.zeros_like(pointnet_batched_result)
        # self.assertTrue(
        #     torch.equal(pointnet_batched_result, expected_pointnet_batched_result)
        # )

    def test_saved_precog_model(self):
        # XXX: Requires Precog encoder to be initialized with no bias.
        SOCIAL_CAPACITY = 5
        NUM_SOCIAL_FEATURES = 4
        NUM_SEEABLE_VEHICLES = 1

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        experiment_path = (
            "logs/experiment-2021.5.4-21:42:46-bdqn-v0:OVERNIGHT_NO_PRECOG_BIAS/"
        )
        model_path = os.path.join(experiment_path, "models/000/800844/")

        agent_spec = make(
            locator = "ultra.baselines.dqn:dqn-v0",
            checkpoint_dir=model_path,
            experiment_dir=experiment_path,
            max_episode_steps=1200,
            agent_id="000",
        )
        agent = agent_spec.build_agent()
        encoder = agent.online_q_network.social_feature_encoder

        # Prepare input.
        seeable_vehicles_array = np.random.uniform(
            low=-1.0, high=1.0, size=(NUM_SEEABLE_VEHICLES, NUM_SOCIAL_FEATURES)
        ).astype(np.float32)
        empty_vehicles_array = np.zeros(
            shape=(SOCIAL_CAPACITY - NUM_SEEABLE_VEHICLES, NUM_SOCIAL_FEATURES),
            dtype=np.float32,
        )
        vehicles_array = np.concatenate((seeable_vehicles_array, empty_vehicles_array), axis=0)
        # TODO: Sort by distance.
        print("vehicles_array:", vehicles_array)
        vehicles_tensor = torch.from_numpy(vehicles_array).unsqueeze(axis=0).to(device)
        print("vehicles_tensor:", vehicles_tensor)

        print("result:", encoder(vehicles_tensor))
        # encoder.eval()
        # print(encoder)
        # print(type(encoder))

    def _generate_random_dataset(self, batch_size, input_size, output_size):
        return (
            np.random.uniform(low=-1.0, high=1.0, size=(batch_size, input_size)).astype(
                np.float32
            ),
            np.random.uniform(
                low=-1.0, high=1.0, size=(batch_size, output_size)
            ).astype(np.float32),
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
