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
"""
Centralized A2C policy
"""
from collections import OrderedDict

import numpy as np
from gym import spaces
from ray.rllib.agents.a3c.a3c_tf_policy import A3CLoss
from ray.rllib.agents.trainer import with_common_config
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.evaluation.postprocessing import Postprocessing, compute_advantages
from ray.rllib.models import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_policy import TFPolicy
from ray.rllib.policy.tf_policy_template import build_tf_policy
from ray.rllib.utils import try_import_tf
from ray.rllib.utils.tf_ops import explained_variance, make_tf_callable

from benchmark.networks import CentralizedActorCriticModel

tf = try_import_tf()


class CentralizedValueMixin:
    def __init__(self: TFPolicy):
        self.compute_central_vf = make_tf_callable(
            self.get_session(), dynamic_shape=True
        )(self.model.central_value_function)


def build_cac_model(
    policy: TFPolicy, obs_space: spaces.Space, action_space: spaces.Space, config
) -> ModelV2:
    policy.model = ModelCatalog.get_model_v2(
        obs_space=obs_space,
        action_space=action_space,
        num_outputs=action_space.n
        if isinstance(action_space, spaces.Discrete)
        else np.product(action_space.shape),
        model_config=config["model"],
        framework="tf",
        default_model=CentralizedActorCriticModel,
        name="cac",
    )

    policy.target_model = ModelCatalog.get_model_v2(
        obs_space=obs_space,
        action_space=action_space,
        num_outputs=np.product(action_space.shape),
        model_config=config["model"],
        framework="tf",
        default_model=CentralizedActorCriticModel,
        name="target_cac",
    )

    return policy.model


def postprocess_trajectory(
    policy: TFPolicy, sample_batch: SampleBatch, other_agent_batches=None, episode=None
):
    last_r = 0.0
    batch_length = len(sample_batch[SampleBatch.CUR_OBS])
    action_preprocessor = policy.model.act_preprocessor
    obs_preprocessor = policy.model.obs_preprocessor

    mean_action = np.zeros((batch_length,) + action_preprocessor.shape)
    own_action = np.zeros((batch_length,) + action_preprocessor.shape)
    own_obs = np.zeros((batch_length,) + obs_preprocessor.shape)

    if policy.loss_initialized():
        sample_batch[SampleBatch.DONES][-1] = 1
        # ordered by agent keys
        other_agent_batches = OrderedDict(other_agent_batches)
        for i, (other_id, (other_policy, batch)) in enumerate(
            other_agent_batches.items()
        ):
            copy_length = min(batch_length, batch[SampleBatch.CUR_OBS].shape[0])

            # TODO(ming): check the action type
            if isinstance(policy.action_space, spaces.Discrete):
                buffer_action = np.eye(action_preprocessor.size)[
                    batch[SampleBatch.ACTIONS][:copy_length]
                ]
            elif isinstance(policy.action_space, spaces.Box):
                buffer_action = batch[SampleBatch.ACTIONS][:copy_length]
            else:
                raise NotImplementedError(
                    f"Do not support such an action space yet:{type(policy.action_space)}"
                )

            mean_action[:copy_length] += buffer_action
        # fill my features to critic_obs_array
        if isinstance(policy.action_space, spaces.Box):
            buffer_action = sample_batch[SampleBatch.ACTIONS]
        elif isinstance(policy.action_space, spaces.Discrete):
            buffer_action = np.eye(action_preprocessor.size)[
                sample_batch[SampleBatch.ACTIONS][:batch_length]
            ]
        else:
            raise NotImplementedError(
                f"Do not support such an action space yte: {type(policy.action_space)}"
            )
        own_action[:batch_length] = buffer_action
        own_obs[:] = sample_batch[SampleBatch.CUR_OBS]
        mean_action /= max(1, len(other_agent_batches))

        sample_batch[CentralizedActorCriticModel.CRITIC_OBS] = np.concatenate(
            [own_obs, own_action, mean_action], axis=-1
        )
        sample_batch[SampleBatch.VF_PREDS] = policy.compute_central_vf(
            sample_batch[CentralizedActorCriticModel.CRITIC_OBS]
        )
    else:
        sample_batch[CentralizedActorCriticModel.CRITIC_OBS] = np.concatenate(
            [own_obs, own_action, mean_action], axis=-1
        )
        sample_batch[SampleBatch.VF_PREDS] = np.zeros_like(
            (batch_length,), dtype=np.float32
        )

    train_batch = compute_advantages(
        sample_batch,
        last_r,
        policy.config["gamma"],
        policy.config["lambda"],
        policy.config["use_gae"],
    )
    return train_batch


def ac_loss_func(policy, model, dist_class, train_batch):
    """Predefined actor-critic loss reuse."""
    logits, _ = model.from_batch(train_batch)
    action_dist = dist_class(logits, model)

    policy.loss = A3CLoss(
        action_dist,
        train_batch[SampleBatch.ACTIONS],
        train_batch[Postprocessing.ADVANTAGES],
        train_batch[Postprocessing.VALUE_TARGETS],
        model.central_value_function(
            train_batch[CentralizedActorCriticModel.CRITIC_OBS]
        ),
        policy.config["vf_loss_coeff"],
        policy.config["entropy_coeff"],
    )

    return policy.loss.total_loss


def central_vf_stats(policy, train_batch, grads):
    # Report the explained variance of the central value function.
    return {
        "vf_explained_var": explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS], policy.central_value_out
        )
    }


def setup_mixins(policy, obs_space, action_space, config):
    CentralizedValueMixin.__init__(policy)


DEFAULT_CONFIG = with_common_config(
    {
        "gamma": 0.95,
        "lambda": 0.0,
        "use_gae": False,
        "vf_loss_coeff": 0.5,
        "entropy_coeff": 0.1,
        "truncate_episodes": True,
    }
)


MFACTFPolicy = build_tf_policy(
    name="MFACTFPolicy",
    loss_fn=ac_loss_func,
    postprocess_fn=postprocess_trajectory,
    before_loss_init=setup_mixins,
    make_model=build_cac_model,
    mixins=[CentralizedValueMixin],
    get_default_config=lambda: DEFAULT_CONFIG,
)


MFACTrainer = build_trainer(
    name="MFAC", default_policy=MFACTFPolicy, default_config=DEFAULT_CONFIG
)
