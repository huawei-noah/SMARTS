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
import numpy as np
import torch

from ultra.baselines.common.replay_buffer import ReplayBuffer
from ultra.baselines.configs import Config


class TD3Config(Config):
    def __init__(self, task):
        super().__init__(task=task)
        self.set_config(
            seed=2,
            social_capacity=5,
            action_size=2,
            social_vehicle_encoder="pointnet_encoder",
            save_codes=[
                "ultra/src/train.py",
                "ultra/baselines/ddpg/config.py",
                "ultra/baselines/ddpg/policy.py",
                "ultra/baselines/ddpg/fc_model.py",
                "ultra/utils/common.py",
                "ultra/src/adapter.py",
            ],
        )
        self.set_config(
            policy_params={
                "state_size": self.state_size,
                "action_size": self.action_size,
                "action_range": np.asarray(
                    [[-1.0, 1.0], [-1.0, 1.0]], dtype=np.float32
                ),
                "state_preprocessor": self.state_preprocessor,
                "update_rate": 5,
                "policy_delay": 2,
                "noise_clip": 0.5,
                "policy_noise": 0.2,
                "warmup": 10000,
                "actor_lr": 1e-4,
                "critic_lr": 1e-3,
                "critic_wd": 0.0,
                "actor_wd": 0.0,
                "critic_tau": 0.01,
                "actor_tau": 0.01,
                "device_name": self.device_name,
                "seed": self.seed,
                "gamma": 0.99,
                "batch_size": 128,
                "sigma": 0.3,
                "theta": 0.15,
                "dt": 1e-2,
                "replay": ReplayBuffer(
                    buffer_size=int(1e6),
                    batch_size=128,
                    state_preprocessor=self.state_preprocessor,
                    device_name=self.device_name,
                ),
                "social_feature_encoder_class": self.social_feature_encoder_class,
                "social_feature_encoder_params": self.social_feature_encoder_params,
            },
        )
