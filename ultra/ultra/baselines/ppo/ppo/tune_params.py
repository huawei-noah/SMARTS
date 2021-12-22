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
from ray import tune

config = {
    "batch_size": 2048,  # size of batch
    "lr": 3e-5,
    "mini_batch_size": 64,  # 64
    "epoch_count": 20,
    "gamma": tune.choice([0.99, 0.999]),  # discounting
    "l": 0.95,  # lambda used in lambda-return
    "eps": 0.2,  # epsilon value used in PPO clipping
    "critic_tau": 1.0,
    "actor_tau": 1.0,
    "entropy_tau": 0.0,
    "hidden_units": 512,
    "seed": 2,
    "logging_freq": 2,
    "social_vehicles": {
        "encoder_key": tune.choice(
            ["no_encoder", "precog_encoder", "pointnet_encoder"]
        ),
        "social_policy_hidden_units": 128,
        "social_policy_init_std": 0.5,
    },
    "action_type": "default_action_continuous",
    "observation_type": "default_observation_vector",
    "reward_type": "default_reward",
}
