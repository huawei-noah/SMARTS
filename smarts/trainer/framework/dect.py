"""Decentralized execution centralized training framework
reference: https://github.com/ray-project/ray/blob/master/rllib/examples/centralized_critic.py
"""

import tensorflow as tf
import numpy as np

from gym import spaces
from collections import namedtuple

from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.policy.tf_policy import TFPolicy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation.postprocessing import compute_advantages
from ray.rllib.utils.tf_ops import make_tf_callable
from ray.rllib.agents.trainer import with_common_config
from ray.rllib.policy import build_tf_policy

from smarts.trainer.utils import Network


OPPONENT_OBS = "opponent_observation_list"
OPPONENT_ACTION = "opponent_action_list"
EXTRA = "extra_list"
FRAMEWORK_NAME = "cc_model"


BatchInfo = namedtuple("BatchInfo", "opponent, extra")
OppoBatchInfo = namedtuple("OppoBatchInfo", "obs_shape, act_shape")
ExtraBatchInfo = namedtuple("ExtraBatchInfo", "shape")


class CentralizedCriticModel(TFModelV2):
    def __init__(
        self,
        obs_space: spaces.Space,
        action_space: spaces.Space,
        num_outputs: spaces.Space,
        model_config: dict,
        name: str,
    ):
        # filter model_config
        extra_config = model_config["custom_options"]
        model_config["custom_options"] = None

        super(CentralizedCriticModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )

        # inner network
        self.model = extra_config["actor"](
            obs_space, action_space, num_outputs, model_config, name
        )
        self.register_variables(self.model.variables())

        # central value function maps {global_observation} -> value_pred
        obs = tf.keras.Input(shape=obs_space.shape, name="obs")

        oppo_obs_spaces = extra_config.get("oppo_obs_spaces", {})
        oppo_act_spaces = extra_config.get("oppo_act_spaces", {})

        self.oppo_ids = [
            policy_id for policy_id in oppo_obs_spaces if policy_id is not name
        ]

        # ordered dict
        concat_oppo_obs_shape = np.concatenate(
            [np.zeros(tuple(v.shape)) for v in oppo_obs_spaces.values()]
        ).shape

        oppo_obs_clus = tf.keras.Input(
            shape=concat_oppo_obs_shape, name="oppo-observation"
        )

        concat_oppo_act_shape = np.concatenate(
            [np.zeros(tuple(v.shape)) for v in oppo_act_spaces.values()]
        ).shape

        oppo_act_clus = tf.keras.Input(shape=concat_oppo_act_shape, name="oppo-action")

        extra_spaces = extra_config.get("extra_spaces", {})
        extra_input = [
            tf.keras.Input(shape=(None,) + v.shape, name="extra-{}-phs".format(k))
            for k, v in extra_spaces.items()
        ]

        inputs = [obs, oppo_obs_clus, oppo_act_clus] + extra_input
        concat_state = tf.keras.layers.Concatenate(axis=-1)(inputs)

        self.central_vf = Network.new_instance_tf_v2(
            concat_state, extra_config["critic"], inputs=inputs
        )
        self.register_variables(self.central_vf.variables)
        self._oppo_act_spaces = oppo_act_spaces
        self._oppo_obs_spaces = oppo_obs_spaces

        self._opponent_info = OppoBatchInfo(
            concat_oppo_obs_shape, concat_oppo_act_shape
        )
        self._extra_info = [ExtraBatchInfo(e) for e in extra_spaces]

    @property
    def info(self):
        return BatchInfo(self._opponent_info, self._extra_info)

    def forward(self, input_dict, state, seq_lens):
        return self.model.forward(input_dict, state, seq_lens)

    def central_value_function(self, obs, oppo_obs, oppo_action, extra=[]):
        # TODO(ming): make inputs as dicts is better
        return tf.reshape(self.central_vf([obs, oppo_obs, oppo_action] + extra), [-1])

    def value_function(self):
        return self.model.value_function()


class CentralizedValueMixin:
    def __init__(self):
        self.compute_central_vf = make_tf_callable(
            self.get_session(), dynamic_shape=True
        )(self.model.central_value_function)


def postprocess_trajectory(
    policy: TFPolicy, sample_batch: SampleBatch, other_agent_batches=None, episode=None
):
    # TODO(ming): only accept trajectory with terminated state, will consider non-terminated state in the future.
    last_r = 0.0
    batch_info = policy.model.info
    oppo_info = batch_info.opponent
    batch_length = sample_batch[SampleBatch.CUR_OBS].shape[0]

    if policy.loss_initialized():
        assert sample_batch["dones"][
            -1
        ], "Not implemented for train_batch_mode=truncate_episodes"
        assert other_agent_batches is not None

        opponent_batch = dict()
        # (other_id, (other_policy, batch))
        for _, (_, batch) in other_agent_batches.items():
            if SampleBatch.CUR_OBS not in opponent_batch:
                opponent_batch[SampleBatch.CUR_OBS] = []
            if SampleBatch.ACTIONS not in opponent_batch:
                opponent_batch[SampleBatch.ACTIONS] = []

            # XXX(why we should do that): in RLlib, batches is corrlated to an episode here,
            # so, tha `batch_length` is no larger than `length_episode` and larger than 0. Consider
            # a 2-agents case, the first dead agent will receive opponent batch whose length is larger than itselves,
            # and another agent will receive opponent batch whose length smaller than its. So, it is necessary to do cutting
            # or fill zeros for agents.
            copy_length = min(batch_length, batch[SampleBatch.CUR_OBS].shape[0])
            tmp_cur_obs = np.zeros(
                (batch_length,) + oppo_info.obs_shape, dtype=np.float32
            )
            tmp_cur_act = np.zeros(
                (batch_length,) + oppo_info.act_shape, dtype=np.float32
            )
            tmp_cur_obs[:copy_length] = batch[SampleBatch.CUR_OBS][:copy_length]
            tmp_cur_act[:copy_length] = batch[SampleBatch.ACTIONS][:copy_length]

            opponent_batch[SampleBatch.CUR_OBS].append(tmp_cur_obs)
            opponent_batch[SampleBatch.ACTIONS].append(tmp_cur_act)

        # also record the opponent obs and actions in the trajectory
        sample_batch[OPPONENT_OBS] = np.hstack(opponent_batch[SampleBatch.CUR_OBS])
        sample_batch[OPPONENT_ACTION] = np.hstack(opponent_batch[SampleBatch.ACTIONS])

        # overwrite default VF prediction with the central vf
        sample_batch[SampleBatch.VF_PREDS] = policy.compute_central_vf(
            sample_batch[SampleBatch.CUR_OBS],
            sample_batch[OPPONENT_OBS],
            sample_batch[OPPONENT_ACTION],
        )
    else:
        sample_batch[OPPONENT_OBS] = np.zeros(
            (sample_batch[SampleBatch.CUR_OBS].shape[0],) + oppo_info.obs_shape,
            dtype=np.float32,
        )
        sample_batch[OPPONENT_ACTION] = np.zeros(
            (sample_batch[SampleBatch.CUR_OBS].shape[0],) + oppo_info.act_shape,
            dtype=np.float32,
        )

        for info in batch_info.extra:
            if sample_batch.get(EXTRA, None) is None:
                sample_batch[EXTRA] = []

            sample_batch[EXTRA].append(
                np.zeros(
                    (sample_batch[SampleBatch.CUR_OBS].shape[0],) + info.shape,
                    dtype=np.float32,
                )
            )

        if sample_batch.get(EXTRA):
            sample_batch[EXTRA] = np.hstack(sample_batch[EXTRA])

        sample_batch[SampleBatch.VF_PREDS] = np.zeros_like(
            (sample_batch[SampleBatch.ACTIONS].shape[0],), dtype=np.float32
        )

    train_batch = compute_advantages(
        sample_batch,
        last_r,
        policy.config["gamma"],
        policy.config["lambda"],
        policy.config["use_gae"],
    )
    return train_batch


ModelCatalog.register_custom_model("cc_model", CentralizedCriticModel)


def setup_mixins(policy, obs_space, action_space, config):
    CentralizedValueMixin.__init__(policy)


DECTPolicy = build_tf_policy(
    name="DECTPolicy",
    loss_fn=None,
    postprocess_fn=postprocess_trajectory,
    before_loss_init=setup_mixins,
    mixins=[CentralizedValueMixin],
    get_default_config=lambda: with_common_config(
        {"gamma": 0.95, "lambda": 0.0, "use_gae": False}
    ),
)
