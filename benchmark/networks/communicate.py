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
import copy
from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf1
from ray.rllib.evaluation.postprocessing import compute_advantages
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_policy import TFPolicy
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.tf_run_builder import TFRunBuilder
from ray.rllib.utils.types import ModelGradients, TensorType

_, tf, tf_version = try_import_tf()


SampleBatch.OTHER_VARIABLES = None
SampleBatch.CONSENSUS_NET = None
SampleBatch.OTHER_GRADIENTS = None


def name_ref(name):
    name = "_".join(name.split("/")[1:])
    name = "_".join(name.split(":"))
    return name


class NetworkedMixin:
    """ Mixin class will be initialized after base class """

    def __init__(self: TFPolicy):
        grads_and_vars = self.init_grads_and_vars()
        self._gamma_grads_ndarray = {k: v[0] for k, v in grads_and_vars.items()}
        self._old_grads_ndarray = copy.copy(self._gamma_grads_ndarray)

    @property
    def gamma_grads_ndarray_dict(self):
        return self._gamma_grads_ndarray

    @property
    def vars_ndarray_dict(self: TFPolicy):
        var_dict = self.model.variables(as_dict=True)
        var_dict = OrderedDict(var_dict)
        vars = var_dict.values()
        res = dict(zip(var_dict.keys(), self._sess.run(list(vars))))
        return res

    def get_pure_var_names(self: TFPolicy):
        raw_var_keys = list(self.model.variables(as_dict=True).keys())
        pure_keys = map(lambda x: name_ref(x), raw_var_keys)
        return list(pure_keys)

    def init_grads_and_vars(self: TFPolicy):
        var_dict = self.model.variables(as_dict=True)
        for key, var_tf in var_dict.items():
            var = self._sess.run(var_tf)
            var_dict[key] = (
                np.zeros(var.shape, dtype=var_tf.dtype.as_numpy_dtype),
                var,
            )

        return var_dict

    def compute_gradients(
        self: TFPolicy, postprocessed_batch: SampleBatch
    ) -> Tuple[ModelGradients, Dict[str, TensorType]]:
        assert self.loss_initialized()
        builder = TFRunBuilder(self._sess, "compute_gradients")
        fetches = self._build_compute_gradients(builder, postprocessed_batch)
        results = builder.get(fetches)
        grads = results[0]

        # ===== update gamma grads
        vars_tf = [v for (g, v) in self._grads_and_vars]
        for var_tf, grad in zip(vars_tf, grads):
            name = name_ref(var_tf.name)
            # (batch, *grad_shape): it is a summation
            gamma_grad_mat = postprocessed_batch[f"gamma_{name}"][0].reshape(grad.shape)

            if self._old_grads_ndarray.get(var_tf.name) is None:
                self._old_grads_ndarray[var_tf.name] = np.zeros_like(grad)

            assert (
                gamma_grad_mat.shape
                == grad.shape
                == self._old_grads_ndarray[var_tf.name].shape
            ), (
                gamma_grad_mat.shape,
                grad.shape,
                self._old_grads_ndarray[var_tf.name].shape,
                var_tf.name,
            )
            self._gamma_grads_ndarray[var_tf.name] = (
                gamma_grad_mat + grad - self._old_grads_ndarray[var_tf.name]
            )
            self._old_grads_ndarray[var_tf.name] = grad
        return results

    def build_apply_op(
        self: TFPolicy,
        optimizer: "tf.keras.optimizers.Optimizer",
        grads_and_vars: List[Tuple[TensorType, TensorType]],
    ) -> "tf.Operation":
        new_grads_and_vars = []
        for grad, var in self._grads_and_vars:
            name = var.name
            name = name_ref(name)
            var_phs = self.get_placeholder(f"var_{name}")
            gamma_grad_phs = self.get_placeholder(f"gamma_{name}")
            new_grad = (
                var
                - tf1.reduce_mean(var_phs, axis=0)  # variable placeholder
                - self.config["lr"] * tf1.reduce_mean(gamma_grad_phs, axis=0)
            )
            assert new_grad.shape == var.shape, (new_grad.shape, var.shape)
            new_grads_and_vars.append((new_grad, var))
        return optimizer.apply_gradients(
            new_grads_and_vars, global_step=tf1.train.get_or_create_global_step()
        )


def postprocess_trajectory(
    policy: TFPolicy, sample_batch: SampleBatch, other_agent_batches=None, episode=None
):
    last_r = 0.0
    batch_length = len(sample_batch[SampleBatch.CUR_OBS])
    var_names = policy.get_pure_var_names()
    other_gradients = {k: [] for k in var_names}
    other_vars = {k: [] for k in var_names}

    if policy.loss_initialized():
        for other_id, (other_policy, batch) in other_agent_batches.items():
            assert isinstance(other_policy, TFPolicy)
            assert isinstance(batch, SampleBatch)

            grads, vars = (
                other_policy.gamma_grads_ndarray_dict,
                other_policy.vars_ndarray_dict,
            )
            for k in grads:
                var = vars[k]
                grad = grads[k]
                name = name_ref(k)

                assert var.shape == grad.shape, (k, var.shape, grad.shape)
                other_gradients[name].append(grad / len(other_agent_batches))
                other_vars[name].append(var / len(other_agent_batches))

        for v in other_vars.values():
            assert len(v) == len(other_agent_batches), (
                len(v),
                len(other_agent_batches),
            )

            # pack other_gradients / other_vars as ndarray objects
        for name, grad_nested_list in other_gradients.items():
            assert len(other_vars[name]) > 0, name
            assert len(grad_nested_list) > 0, name
            var_nested = np.sum(other_vars[name], axis=0)
            grad_nested = np.sum(grad_nested_list, axis=0)

            assert var_nested.shape == other_vars[name][0].shape, (
                var_nested.shape,
                other_vars[name][0].shape,
            )

            assert grad_nested.shape == var_nested.shape, (
                grad_nested.shape,
                var_nested.shape,
            )
            reshape = (batch_length,) + tuple([1] * len(grad_nested.shape))

            sample_batch[f"gamma_{name}"] = np.tile(grad_nested, reshape)
            sample_batch[f"var_{name}"] = np.tile(var_nested, reshape)
            assert (
                sample_batch[f"gamma_{name}"].shape == sample_batch[f"var_{name}"].shape
            ), (sample_batch[f"gamma_{name}"].shape, sample_batch[f"var_{name}"].shape)
    else:
        grads_and_vars = policy.init_grads_and_vars()
        for k, (grad, var) in grads_and_vars.items():
            name = name_ref(k)
            assert other_gradients.get(name, None) is not None, name
            other_gradients[name].append(grad)
            other_vars[name].append(var)
            sample_batch[f"gamma_{name}"] = np.zeros((batch_length,) + grad.shape)
            sample_batch[f"var_{name}"] = np.tile(
                var, (batch_length,) + tuple([1] * len(var.shape))
            )
            assert (
                sample_batch[f"gamma_{name}"].shape == sample_batch[f"var_{name}"].shape
            ), (sample_batch[f"gamma_{name}"].shape, sample_batch[f"var_{name}"].shape)

    train_batch = compute_advantages(
        sample_batch,
        last_r,
        policy.config.get("gamma", 0.9),
        policy.config.get("lambda", 1.0),
        policy.config.get("use_gae", False),
        policy.config.get("use_critic", False),
    )

    return train_batch
