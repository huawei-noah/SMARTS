.. _rllib_in_smarts:

RLlib
=====

**RLlib** is an open-source library for reinforcement learning that offers both high scalability and a unified API for a variety
of applications. RLlib natively supports TensorFlow, TensorFlow Eager, and PyTorch. Most of its internals are agnostic to such
deep learning frameworks.

====================
Recommend Read First
====================

RLlib is implemented on top of Ray. Ray is a distributed computing framework specifically designed with RL in mind. There are
many docs about Ray and RLlib. We recommend to read the following pages first,

- `RLlib in 60 seconds <https://docs.ray.io/en/latest/rllib.html#rllib-in-60-seconds>`_: Getting started with RLlib.
- `Common Parameters <https://docs.ray.io/en/latest/rllib-training.html#common-parameters>`_: Common `tune` configs.
- `Basic Python API <https://docs.ray.io/en/latest/rllib-training.html#basic-python-api>`_: Basic `tune.run` function.
- `Callbacks and Custom Metrics <https://docs.ray.io/en/latest/rllib-training.html#callbacks-and-custom-metrics>`_: About callbacks and metrics.
- `Visualizing Custom Metrics <https://docs.ray.io/en/latest/rllib-training.html#visualizing-custom-metrics>`_: How to use TensorBoard to visualize metrics.
- `Built-in Models and Preprocessors <https://docs.ray.io/en/latest/rllib-models.html#default-behaviours>`_: Built-in preprocessor, including how to deal with different observation spaces.
- `Proximal Policy Optimization (PPO) <https://docs.ray.io/en/latest/rllib-algorithms.html#proximal-policy-optimization-ppo>`_: RLlib PPO implementation and PPO parameters.
- `PopulationBasedTraining <https://docs.ray.io/en/latest/tune/api_docs/schedulers.html#populationbasedtraining>`_: Population Based Training algorithm and examples.
- `RLlib Examples <https://docs.ray.io/en/latest/rllib-examples.html>`_: Get to know RLlib quickly through examples.

=================
SMARTS RLlib Tips
=================

Resume or continue training
---------------------------

If you want to continue an aborted experiemnt. you can set `resume=True` in `tune.run`. But note that`resume=True` will continue to use the same configuration as was set in the original experiment.
To make changes to a started experiment, you can edit the latest experiment file in `~/ray_results/rllib_example`.

Or if you want to start a new experiment but train from an existing checkpoint, you can set `restore=checkpoint_path` in `tune.run`.
