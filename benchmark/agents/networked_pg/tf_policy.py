# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
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
from ray.rllib.agents.pg.pg import DEFAULT_CONFIG as PG_DEFAULT_CONFIG
from ray.rllib.agents.pg.pg_tf_policy import pg_tf_loss
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.policy.tf_policy_template import build_tf_policy

from benchmark.networks.communicate import NetworkedMixin, postprocess_trajectory


def networked_pg_loss(policy, model, dist_class, train_batch):
    # make gradients accessed
    for k in train_batch.keys():
        if "var" in k or "gamma" in k:
            _ = train_batch[k].shape

    return pg_tf_loss(policy, model, dist_class, train_batch)


def setupmixin(policy, obs_space, action_space, config):
    NetworkedMixin.__init__(policy)


NetworkedPG = build_tf_policy(
    name="NetworkedPG",
    get_default_config=lambda: PG_DEFAULT_CONFIG,
    postprocess_fn=postprocess_trajectory,
    loss_fn=networked_pg_loss,
    mixins=[NetworkedMixin],
    after_init=setupmixin,
)


NetworkedPGTrainer = build_trainer(
    name="NetworkedPGTrainer",
    default_policy=NetworkedPG,
)
