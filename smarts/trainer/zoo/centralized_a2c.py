"""
Centralized A2C algorithm settings.
"""

from collections import namedtuple

from ray.rllib.agents.a3c.a3c_tf_policy import A3CLoss
from ray.rllib.agents.trainer import with_common_config
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils import try_import_tf
from ray.rllib.utils.explained_variance import explained_variance

from smarts.trainer.framework.dect import DECTPolicy
from smarts.trainer.framework.dect import OPPONENT_OBS, OPPONENT_ACTION

tf = try_import_tf()


meta = namedtuple("Meta", "policy, trainer")


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
            train_batch[SampleBatch.CUR_OBS],
            train_batch[OPPONENT_OBS],
            train_batch[OPPONENT_ACTION],
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


# customize your policy
CA2CPolicy = DECTPolicy.with_updates(
    name="CA2CPolicy",
    loss_fn=ac_loss_func,
    get_default_config=lambda: with_common_config(
        {
            "gamma": 0.95,
            "lambda": 0.0,
            "use_gae": False,
            "vf_loss_coeff": 0.1,
            "entropy_coeff": 0.1,
        }
    ),
)


# define your trainer
CA2CTrainer = build_trainer(
    name="CA2CTrainer",
    default_policy=CA2CPolicy,
    default_config=with_common_config({"truncate_episodes": True}),
)

META = meta(CA2CPolicy, CA2CTrainer)
