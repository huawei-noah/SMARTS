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
import random, copy
import torch
from ultra import adapters
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

    def __init__(self, buffer_size, device):
        self._buffer_size = buffer_size
        self._memory = deque(maxlen=self._buffer_size)
        self._device = device

    def add(
        self,
        state,
        action,
        reward,
        next_state,
        done,
        prev_action,
        others=None,
    ):
        raise NotImplementedError

    @staticmethod
    def make_states_batch(states, device):
        raise NotImplementedError

    def __len__(self):
        return len(self._memory)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        state, action, reward, next_state, done, others = tuple(self._memory[index])
        return state, action, reward, next_state, done, others


class DefaultVectorReplayBufferDataset(ReplayBufferDataset):
    def __init__(self, buffer_size, device):
        super(DefaultVectorReplayBufferDataset, self).__init__(buffer_size, device)

    def add(
        self,
        state,
        action,
        reward,
        next_state,
        done,
        prev_action,
        others=None,
    ):
        if others is None:
            others = {}
        # dereference the states
        state = copy.deepcopy(state)
        next_state = copy.deepcopy(next_state)
        state["low_dim_states"] = np.float32(
            np.append(state["low_dim_states"], prev_action)
        )
        state["low_dim_states"] = torch.from_numpy(state["low_dim_states"]).to(
            self._device
        )
        state["social_vehicles"] = torch.from_numpy(state["social_vehicles"]).to(
            self._device
        )

        next_state["low_dim_states"] = np.float32(
            np.append(next_state["low_dim_states"], action)
        )
        next_state["social_vehicles"] = torch.from_numpy(
            next_state["social_vehicles"]
        ).to(self._device)
        next_state["low_dim_states"] = torch.from_numpy(
            next_state["low_dim_states"]
        ).to(self._device)

        action = np.asarray([action]) if not isinstance(action, Iterable) else action
        action = torch.from_numpy(action).float()
        reward = torch.from_numpy(np.asarray([reward])).float()
        done = torch.from_numpy(np.asarray([done])).float()
        new_experience = Transition(state, action, reward, next_state, done, others)
        self._memory.append(new_experience)

    @staticmethod
    def make_states_batch(states, device):
        # image_keys = states[0]["images"].keys()
        # images = {}
        # for k in image_keys:
        #     _images = torch.cat([e[k] for e in states], dim=0).float().to(device)
        #     _images = normalize_im(_images)
        #     images[k] = _images
        low_dim_states_batch = (
            torch.cat([e["low_dim_states"] for e in states], dim=0).float().to(device)
        )
        if "social_vehicles" in states[0]:
            social_vehicles_batch = [
                e["social_vehicles"][0].float().to(device) for e in states
            ]
        else:
            social_vehicles_batch = False
        out = {
            # "images": images,
            "low_dim_states": low_dim_states_batch,
            "social_vehicles": social_vehicles_batch,
        }
        return out


class DefaultImageReplayBufferDataset(ReplayBufferDataset):
    def __init__(self, buffer_size, device):
        super(DefaultImageReplayBufferDataset, self).__init__(buffer_size, device)

    def add(
        self,
        state,
        action,
        reward,
        next_state,
        done,
        prev_action,
        others=None,
    ):
        if others is None:
            others = {}

        # Dereference the states.
        state = copy.deepcopy(state)
        next_state = copy.deepcopy(next_state)

        state = torch.from_numpy(state).to(self._device)
        next_state = torch.from_numpy(next_state).to(self._device)
        next_state = next_state[-1].unsqueeze(dim=0)  # Keep only the newest frame.
        action = np.asarray([action]) if not isinstance(action, Iterable) else action
        action = torch.from_numpy(action).float()
        reward = torch.from_numpy(np.asarray([reward])).float()
        done = torch.from_numpy(np.asarray([done])).float()

        new_experience = Transition(state, action, reward, next_state, done, others)

        self._memory.append(new_experience)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        state, action, reward, next_state, done, others = tuple(self._memory[index])
        next_state = torch.cat((state[1:], next_state))  # Reattach the previous frames.
        return state, action, reward, next_state, done, others

    @staticmethod
    def make_states_batch(states, device):
        return torch.cat(states, dim=0).float().to(device)


class ReplayBuffer:
    def __init__(
        self,
        buffer_size,
        batch_size,
        observation_type,
        device_name,
        pin_memory=False,
        num_workers=0,
    ):
        if observation_type == adapters.AdapterType.DefaultObservationVector:
            self.replay_buffer_dataset = DefaultVectorReplayBufferDataset(
                buffer_size, device=None
            )
        elif observation_type == adapters.AdapterType.DefaultObservationImage:
            self.replay_buffer_dataset = DefaultImageReplayBufferDataset(
                buffer_size, device=None
            )
        else:
            raise Exception(f"'{observation_type}' does not have a DataSet.")

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

    def sample(self, device=None):
        device = device if device else self.storage_device
        batch = list(iter(self.data_loader))
        states, actions, rewards, next_states, dones, others = zip(*batch)
        states = self.replay_buffer_dataset.make_states_batch(states, device)
        next_states = self.replay_buffer_dataset.make_states_batch(next_states, device)
        actions = torch.cat(actions, dim=0).float().to(device)
        rewards = torch.cat(rewards, dim=0).float().to(device)
        dones = torch.cat(dones, dim=0).float().to(device)
        return states, actions, rewards, next_states, dones, others
