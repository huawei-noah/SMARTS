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
# Do not make any change to this file when merging. Just use my version.
from collections import deque, namedtuple
import numpy as np
import random
import torch
from ultra.utils.common import normalize_im
from collections.abc import Iterable

from torch.utils.data import Dataset, Sampler, DataLoader

Transition = namedtuple(
    "Transition",
    field_names=["state", "action", "reward", "next_state", "done", "others"],
    # others may contain importance sampling ratio, GVF rewards,... etc
    defaults=(None,) * 6,
)


class RandomRLSampler(Sampler):
    def __init__(self, data_source, batch_size):
        self.datasource = data_source
        self.batch_size = batch_size

    def __len__(self):
        return len(self.datasource)

    def __iter__(self):
        n = len(self.datasource)
        return iter(torch.randperm(n).tolist()[0 : self.batch_size])


class ReplayBufferDataset(Dataset):
    cpu = torch.device("cpu")

    def __init__(self, buffer_size, state_preprocessor, device):
        self.buffer_size = buffer_size
        self.memory = deque(maxlen=self.buffer_size)
        self.state_preprocessor = state_preprocessor
        self.device = device

    def add(
        self,
        state,
        action,
        reward,
        next_state,
        done,
        social_capacity,
        observation_num_lookahead,
        social_vehicle_config,
        prev_action,
        others=None,
    ):
        if others is None:
            others = {}
        state = self.state_preprocessor(
            state,
            normalize=False,
            unsqueeze=False,
            device=self.device,
            social_capacity=social_capacity,
            observation_num_lookahead=observation_num_lookahead,
            social_vehicle_config=social_vehicle_config,
            prev_action=prev_action,
        )
        next_state = self.state_preprocessor(
            next_state,
            normalize=False,
            unsqueeze=False,
            device=self.device,
            social_capacity=social_capacity,
            observation_num_lookahead=observation_num_lookahead,
            social_vehicle_config=social_vehicle_config,
            prev_action=action,
        )

        action = np.asarray([action]) if not isinstance(action, Iterable) else action
        action = torch.from_numpy(action).float()
        reward = torch.from_numpy(np.asarray([reward])).float()
        done = torch.from_numpy(np.asarray([done])).float()
        new_experience = Transition(state, action, reward, next_state, done, others)
        self.memory.append(new_experience)

    def __len__(self):
        return len(self.memory)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        state, action, reward, next_state, done, others = tuple(self.memory[idx])
        return state, action, reward, next_state, done, others


class ReplayBuffer:
    def __init__(
        self,
        buffer_size,
        batch_size,
        state_preprocessor,
        device_name,
        pin_memory=False,
        num_workers=0,
    ):
        self.replay_buffer_dataset = ReplayBufferDataset(
            buffer_size, state_preprocessor, device=None
        )
        self.sampler = RandomRLSampler(self.replay_buffer_dataset, batch_size)
        self.data_loader = DataLoader(
            self.replay_buffer_dataset,
            sampler=self.sampler,
            pin_memory=pin_memory,
            num_workers=num_workers,
        )
        self.storage_device = torch.device(device_name)

    def add(self, *args, **kwargs):
        self.replay_buffer_dataset.add(*args, **kwargs)

    def __len__(self):
        return len(self.replay_buffer_dataset)

    def __getitem__(self, idx):
        return self.replay_buffer_dataset[idx]

    def make_state_from_dict(self, states, device):
        image_keys = states[0]["images"].keys()
        images = {}
        for k in image_keys:
            _images = torch.cat([e[k] for e in states], dim=0).float().to(device)
            _images = normalize_im(_images)
            images[k] = _images
        low_dim_states = (
            torch.cat([e["low_dim_states"] for e in states], dim=0).float().to(device)
        )
        if "social_vehicles" in states[0]:
            social_vehicles = [
                e["social_vehicles"][0].float().to(device) for e in states
            ]
        else:
            social_vehicles = False
        out = {
            "images": images,
            "low_dim_states": low_dim_states,
            "social_vehicles": social_vehicles,
        }
        return out

    def sample(self, device=None):
        device = device if device else self.storage_device
        batch = list(iter(self.data_loader))
        states, actions, rewards, next_states, dones, others = zip(*batch)
        states = self.make_state_from_dict(states, device)
        next_states = self.make_state_from_dict(next_states, device)
        actions = torch.cat(actions, dim=0).float().to(device)
        rewards = torch.cat(rewards, dim=0).float().to(device)
        dones = torch.cat(dones, dim=0).float().to(device)
        return states, actions, rewards, next_states, dones, others
