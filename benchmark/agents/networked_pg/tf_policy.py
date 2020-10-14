from ray.rllib.agents.pg.pg_tf_policy import pg_tf_loss
from ray.rllib.agents.pg.pg import DEFAULT_CONFIG as PG_DEFAULT_CONFIG

from ray.rllib.policy.tf_policy_template import build_tf_policy
from ray.rllib.agents.trainer_template import build_trainer

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
    name="NetworkedPGTrainer", default_policy=NetworkedPG,
)
