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
from smarts.core.agent import Agent
from ultra.utils.common import merge_discrete_action_spaces, to_3d_action, to_2d_action
import pathlib, os, copy
import ultra.adapters as adapters
from ultra.baselines.dqn.dqn.explore import EpsilonExplore
from ultra.baselines.dqn.dqn.network import DQNCNN, DQNWithSocialEncoder
from ultra.baselines.common.replay_buffer import ReplayBuffer
from ultra.baselines.common.social_vehicle_config import get_social_vehicle_configs
from ultra.baselines.common.yaml_loader import load_yaml


class DQNPolicy(Agent):
    lane_actions = ["keep_lane", "slow_down", "change_lane_left", "change_lane_right"]

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

        if self.action_type == adapters.AdapterType.DefaultActionContinuous:
            discrete_action_spaces = [
                np.asarray([-0.25, 0.0, 0.5, 0.75, 1.0]),
                np.asarray(
                    [-1.0, -0.75, -0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
                ),
            ]
            self.index2actions = [
                merge_discrete_action_spaces([discrete_action_space])[0]
                for discrete_action_space in discrete_action_spaces
            ]
            self.action2indexs = [
                merge_discrete_action_spaces([discrete_action_space])[1]
                for discrete_action_space in discrete_action_spaces
            ]
            self.merge_action_spaces = 0
            self.num_actions = [
                len(discrete_action_space)
                for discrete_action_space in discrete_action_spaces
            ]
            self.action_size = 2
            self.to_real_action = to_3d_action
        elif self.action_type == adapters.AdapterType.DefaultActionDiscrete:
            discrete_action_spaces = [[0], [1], [2], [3]]
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
                f"DQN baseline does not support the '{self.action_type}' action type."
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

    def lane_action_to_index(self, state):
        state = state.copy()
        if (
            len(state["action"]) == 3
            and (state["action"] == np.asarray([0, 0, 0])).all()
        ):  # initial action
            state["action"] = np.asarray([0])
        else:
            state["action"] = self.lane_actions.index(state["action"])
        return state

    def reset(self):
        self.eps_throttles = []
        self.eps_steers = []
        self.eps_step = 0
        self.current_sticky = 0

    def soft_update(self, target, src, tau):
        for target_param, param in zip(target.parameters(), src.parameters()):
            target_param.detach_()
            target_param.copy_(target_param * (1.0 - tau) + param * tau)

    def update_target_network(self):
        self.target_q_network.load_state_dict(self.online_q_network.state_dict().copy())

    def act(self, *args, **kwargs):
        if self.current_sticky == 0:
            self.action = self._act(*args, **kwargs)
        self.current_sticky = (self.current_sticky + 1) % self.sticky_actions
        self.current_iteration += 1
        return self.to_real_action(self.action)

    def _act(self, state, explore=True):
        epsilon = self.epsilon_obj.get_epsilon()
        if not explore or np.random.rand() > epsilon:
            state = copy.deepcopy(state)
            if self.observation_type == adapters.AdapterType.DefaultObservationVector:
                # Default vector observation type.
                state["low_dim_states"] = np.float32(
                    np.append(state["low_dim_states"], self.prev_action)
                )
                state["social_vehicles"] = (
                    torch.from_numpy(state["social_vehicles"])
                    .unsqueeze(0)
                    .to(self.device)
                )
                state["low_dim_states"] = (
                    torch.from_numpy(state["low_dim_states"])
                    .unsqueeze(0)
                    .to(self.device)
                )
            else:
                # Default image observation type.
                state = torch.from_numpy(state).unsqueeze(0).to(self.device)
            self.online_q_network.eval()
            with torch.no_grad():
                qs = self.online_q_network(state)
            qs = [q.data.cpu().numpy().flatten() for q in qs]
            # out_str = " || ".join(
            #     [
            #         " ".join(
            #             [
            #                 "{}: {:.4f}".format(index2action[j], q[j])
            #                 for j in range(num_action)
            #             ]
            #         )
            #         for index2action, q, num_action in zip(
            #             self.index2actions, qs, self.num_actions
            #         )
            #     ]
            # )
            # print(out_str)
            inds = [np.argmax(q) for q in qs]
        else:
            inds = [np.random.randint(num_action) for num_action in self.num_actions]
        action = []
        for j, ind in enumerate(inds):
            action.extend(self.index2actions[j][ind])
        self.epsilon_obj.step()
        self.eps_step += 1
        action = np.asarray(action)
        return action

    def save(self, model_dir):
        model_dir = pathlib.Path(model_dir)
        torch.save(self.online_q_network.state_dict(), model_dir / "online.pth")
        torch.save(self.target_q_network.state_dict(), model_dir / "target.pth")

    def load(self, model_dir, cpu=False):
        model_dir = pathlib.Path(model_dir)
        print("loading from :", model_dir)

        map_location = None
        if cpu:
            map_location = torch.device("cpu")
        self.online_q_network.load_state_dict(
            torch.load(model_dir / "online.pth", map_location=map_location)
        )
        self.target_q_network.load_state_dict(
            torch.load(model_dir / "target.pth", map_location=map_location)
        )
        print("Model loaded")

    def step(self, state, action, reward, next_state, done, info, others=None):
        # dont treat timeout as done equal to True
        max_steps_reached = info["logs"]["events"].reached_max_episode_steps
        if max_steps_reached:
            done = False
        if self.action_type == adapters.AdapterType.DefaultActionContinuous:
            action = to_2d_action(action)
            _action = (
                [[e] for e in action]
                if not self.merge_action_spaces
                else [action.tolist()]
            )
            action_index = np.asarray(
                [
                    action2index[str(e)]
                    for action2index, e in zip(self.action2indexs, _action)
                ]
            )
        else:
            action_index = self.lane_actions.index(action)
            action = action_index
        self.replay.add(
            state=state,
            action=action_index,
            reward=reward,
            next_state=next_state,
            done=done,
            others=others,
            prev_action=self.prev_action,
        )
        if (
            self.step_count % self.train_step == 0
            and len(self.replay) >= self.batch_size
            and (self.warmup is None or len(self.replay) >= self.warmup)
        ):
            out = self.learn()
            self.update_count += 1
        else:
            out = {}

        if self.target_update > 1 and self.step_count % self.target_update == 0:
            self.update_target_network()
        elif self.target_update < 1.0:
            self.soft_update(
                self.target_q_network, self.online_q_network, self.target_update
            )
        self.step_count += 1
        self.prev_action = action

        return out

    def learn(self):
        states, actions, rewards, next_states, dones, others = self.replay.sample(
            device=self.device
        )
        if not self.merge_action_spaces:
            actions = torch.chunk(actions, len(self.num_actions), -1)
        else:
            actions = [actions]

        self.target_q_network.eval()
        with torch.no_grad():
            qs_next_target = self.target_q_network(next_states)

        if self.use_ddqn:
            self.online_q_network.eval()
            with torch.no_grad():
                qs_next_online = self.online_q_network(next_states)
            next_actions = [
                torch.argmax(q_next_online, dim=1, keepdim=True)
                for q_next_online in qs_next_online
            ]
        else:
            next_actions = [
                torch.argmax(q_next_target, dim=1, keepdim=True)
                for q_next_target in qs_next_target
            ]

        qs_next_target = [
            torch.gather(q_next_target, 1, next_action)
            for q_next_target, next_action in zip(qs_next_target, next_actions)
        ]

        self.online_q_network.train()
        qs, aux_losses = self.online_q_network(states, training=True)
        qs = [torch.gather(q, 1, action.long()) for q, action in zip(qs, actions)]
        qs_target_value = [
            rewards + self.gamma * (1 - dones) * q_next_target
            for q_next_target in qs_next_target
        ]
        td_loss = [
            self.loss_func(q, q_target_value).mean()
            for q, q_target_value in zip(qs, qs_target_value)
        ]
        mean_td_loss = sum(td_loss) / len(td_loss)

        loss = mean_td_loss + sum(
            [e["value"] * e["weight"] for e in aux_losses.values()]
        )

        self.optimizers.zero_grad()
        loss.backward()
        self.optimizers.step()

        out = {}
        out.update(
            {
                "loss/td{}".format(j): {
                    "type": "scalar",
                    "data": td_loss[j].data.cpu().numpy(),
                    "freq": 10,
                }
                for j in range(len(td_loss))
            }
        )
        out.update(
            {
                "loss/{}".format(k): {
                    "type": "scalar",
                    "data": v["value"],  # .detach().cpu().numpy(),
                    "freq": 10,
                }
                for k, v in aux_losses.items()
            }
        )
        out.update({"loss/all": {"type": "scalar", "data": loss, "freq": 10}})

        self.num_updates += 1
        return out
