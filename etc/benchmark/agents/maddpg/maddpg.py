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
"""Contributed port of MADDPG from OpenAI baselines.

Reference: https://github.com/ray-project/ray/blob/master/rllib/contrib/maddpg/maddpg.py
"""

import logging

from ray.rllib.agents.dqn.dqn import GenericOffPolicyTrainer
from ray.rllib.agents.trainer import COMMON_CONFIG, with_common_config
from ray.rllib.policy.sample_batch import MultiAgentBatch, SampleBatch
from ray.rllib.utils import merge_dicts

from benchmark.agents.maddpg.tf_policy import MADDPG2TFPolicy

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# yapf: disable
# __sphinx_doc_begin__
DEFAULT_CONFIG = with_common_config({
    # === Framework to run the algorithm ===
    "framework": "tf",

    # === Settings for each individual policy ===
    # ID of the agent controlled by this policy
    "agent_id": None,
    # Use a local critic for this policy.
    "use_local_critic": False,

    # === Evaluation ===
    # Evaluation interval
    "evaluation_interval": None,
    # Number of episodes to run per evaluation period.
    "evaluation_num_episodes": 10,

    # === Model ===
    # Apply a state preprocessor with spec given by the "model" config option
    # (like other RL algorithms). This is mostly useful if you have a weird
    # observation shape, like an image. Disabled by default.
    "use_state_preprocessor": False,
    # Postprocess the policy network model output with these hidden layers. If
    # use_state_preprocessor is False, then these will be the *only* hidden
    # layers in the network.
    "actor_hiddens": [64, 64],
    # Hidden layers activation of the postprocessing stage of the policy
    # network
    "actor_hidden_activation": "relu",
    # Postprocess the critic network model output with these hidden layers;
    # again, if use_state_preprocessor is True, then the state will be
    # preprocessed by the model specified with the "model" config option first.
    "critic_hiddens": [64, 64],
    # Hidden layers activation of the postprocessing state of the critic.
    "critic_hidden_activation": "relu",
    # N-step Q learning
    "n_step": 1,

    # === Replay buffer ===
    # Size of the replay buffer. Note that if async_updates is set, then
    # each worker will have a replay buffer of this size.
    "buffer_size": int(1e6),
    # Observation compression. Note that compression makes simulation slow in
    # MPE.
    "compress_observations": False,
    # If set, this will fix the ratio of replayed from a buffer and learned on
    # timesteps to sampled from an environment and stored in the replay buffer
    # timesteps. Otherwise, the replay will proceed at the native ratio
    # determined by (train_batch_size / rollout_fragment_length).
    "training_intensity": None,
    # Force lockstep replay mode for MADDPG.
    "multiagent": merge_dicts(COMMON_CONFIG["multiagent"], {
        "replay_mode": "lockstep"  # "independent",  # replay mode has two options, they are lockstep and independent, for smarts, it must be independent
    }),

    # === Optimization ===
    # Learning rate for the critic (Q-function) optimizer.
    "critic_lr": 1e-3,
    # Learning rate for the actor (policy) optimizer.
    "actor_lr": 1e-4,
    # Update the target network every `target_network_update_freq` steps.
    "target_network_update_freq": 0,
    # Update the target by \tau * policy + (1-\tau) * target_policy
    "tau": 0.01,
    # Weights for feature regularization for the actor
    "actor_feature_reg": 0.001,
    # If not None, clip gradients during optimization at this value
    "grad_norm_clipping": 0.5,
    # How many steps of the model to sample before learning starts.
    "learning_starts": 1024 * 25,
    # Update the replay buffer with this many samples at once. Note that this
    # setting applies per-worker if num_workers > 1.
    "rollout_fragment_length": 100,
    # Size of a batched sampled from replay buffer for training. Note that
    # if async_updates is set, then each worker returns gradients for a
    # batch of this size.
    "train_batch_size": 1024,
    # Number of env steps to optimize for before returning
    "timesteps_per_iteration": 0,

    # === Parallelism ===
    # Number of workers for collecting samples with. This only makes sense
    # to increase if your environment is particularly slow to sample, or if
    # you're using the Async or Ape-X optimizers.
    "num_workers": 1,
    # Prevent iterations from going lower than this time span
    "min_iter_time_s": 0,
})
# __sphinx_doc_end__
# yapf: enable


def before_learn_on_batch(
    multi_agent_batch: MultiAgentBatch, policies, train_batch_size
):
    samples = {}

    # Modify keys.
    for pid, p in policies.items():
        i = p.agent_idx
        keys = multi_agent_batch.policy_batches[pid].data.keys()
        keys = ["_".join([k, str(i)]) for k in keys]
        samples.update(
            dict(zip(keys, multi_agent_batch.policy_batches[pid].data.values()))
        )

    # Make ops and feed_dict to get "new_obs" from target action sampler.
    new_obs_ph_n = [p.new_obs_ph for p in policies.values()]
    new_obs_n = list()
    for k, v in samples.items():
        if "new_obs" in k:
            new_obs_n.append(v)

    # target_act_sampler_n = [p.target_act_sampler for p in policies.values()]
    feed_dict = dict(zip(new_obs_ph_n, new_obs_n))
    new_act_n = [p.sess.run(p.target_act_sampler, feed_dict) for p in policies.values()]
    samples.update(
        {"new_actions_%d" % i: new_act for i, new_act in enumerate(new_act_n)}
    )

    # Share samples among agents.
    policy_batches = {pid: SampleBatch(samples) for pid in policies.keys()}
    return MultiAgentBatch(policy_batches, train_batch_size)


def add_maddpg_postprocessing(config):
    """Add the before learn on batch hook.

    This hook is called explicitly prior to TrainOneStep() in the execution
    setups for DQN and APEX.
    """

    def f(batch, workers, config):
        policies = dict(
            workers.local_worker().foreach_trainable_policy(lambda p, i: (i, p))
        )
        return before_learn_on_batch(batch, policies, config["train_batch_size"])

    config["before_learn_on_batch"] = f
    return config


MADDPGTrainer = GenericOffPolicyTrainer.with_updates(
    name="MADDPG2",
    default_config=DEFAULT_CONFIG,
    default_policy=MADDPG2TFPolicy,
    get_policy_class=None,
    validate_config=add_maddpg_postprocessing,
)
