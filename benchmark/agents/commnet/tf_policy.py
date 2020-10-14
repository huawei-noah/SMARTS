import gym
import ray
import numpy as np

from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.agents.pg.pg import DEFAULT_CONFIG
from ray.rllib.policy.tf_policy import TFPolicy
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.tf_policy_template import build_tf_policy
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.evaluation.postprocessing import Postprocessing, compute_advantages
from benchmark.networks import CommNet
from ray.rllib.utils.framework import try_import_tf

# import tensorflow as tf

tf1, tf, tfv = try_import_tf()


def build_commnet(
    policy: TFPolicy,
    obs_space: gym.spaces.Space,
    action_space: gym.spaces.Space,
    config,
) -> ModelV2:
    assert isinstance(action_space, gym.spaces.Tuple)

    unit_action_space = action_space.spaces[0]
    unit_action_dim = (
        unit_action_space.n
        if isinstance(unit_action_space, gym.spaces.Discrete)
        else np.product(unit_action_space.shape)
    )

    policy.model = ModelCatalog.get_model_v2(
        obs_space=obs_space,
        action_space=action_space,
        num_outputs=unit_action_dim * len(action_space.spaces),
        model_config=config["model"],
        framework="tf",
        default_model=CommNet,
        name="commnet",
    )

    return policy.model


def postprocess_trajectory(
    policy: TFPolicy, sample_batch: SampleBatch, other_agent_batches=None, episode=None
):
    last_r = 0.0

    train_batch = compute_advantages(
        sample_batch, last_r, use_gae=False, use_critic=False,
    )
    return train_batch


def loss_func(policy, model, dist_class, train_batch):
    logits, _ = policy.model.from_batch(train_batch)
    action_dist = dist_class(logits, model)

    return -tf.reduce_mean(
        action_dist.logp(train_batch[SampleBatch.ACTIONS])
        * tf.cast(train_batch[Postprocessing.ADVANTAGES], dtype=tf.float32)
    )


CommNetTFPolicy = build_tf_policy(
    name="CommNetTFPolicy",
    get_default_config=lambda: DEFAULT_CONFIG,
    postprocess_fn=postprocess_trajectory,
    loss_fn=loss_func,
    make_model=build_commnet,
)

CommNetTrainer = build_trainer(
    name="CommNetTrainer", default_policy=CommNetTFPolicy, default_config=DEFAULT_CONFIG
)
