import tensorflow as tf

from typing import Sequence, NamedTuple, Dict, TypeVar, Callable


class LayerConfig(NamedTuple):
    func: Callable
    kwargs: Dict[str, TypeVar("T")]


class Network:
    @staticmethod
    def new_instance_tf_v2(state, config: Sequence[LayerConfig], inputs: Sequence):
        pre_h = state

        for layerConfig in config:
            h = layerConfig.func(**layerConfig.kwargs)(pre_h)
            pre_h = h

        out = pre_h

        return tf.keras.Model(inputs=inputs, outputs=out)

    @staticmethod
    def new_instance_tf_v1(state, config: Sequence[LayerConfig], inputs: Sequence):
        raise NotImplementedError

    @staticmethod
    def new_instance_torch(state, config: Sequence[LayerConfig], inputs: Sequence):
        raise NotImplementedError


def pack_custom_options_with_opponent(
    agent_id, opponent_ids, options, oppo_obs_pre, oppo_action_pre
):
    new_options = dict(
        options,
        **{
            "oppo_obs_spaces": dict(
                zip(opponent_ids, [oppo_obs_pre.observation_space] * len(opponent_ids))
            ),
            "oppo_act_spaces": dict(
                zip(
                    opponent_ids,
                    [oppo_action_pre.observation_space] * len(opponent_ids),
                )
            ),
            "agent_id": agent_id,
        },
    )

    return new_options
