.. _rllib:


RLlib
=====

**RLlib** is an open-source library for reinforcement learning that offers both high scalability and a unified API for a variety of applications. ``RLlib`` natively supports ``TensorFlow``, ``TensorFlow Eager``, and ``PyTorch``. Most of its internals are agnostic to such deep learning frameworks.

SMARTS contains two examples using `Proximal Policy Optimization (PPO) <https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#ppo>`_.

#. Policy gradient

    + script: :examples:`e12_rllib/ppo_example.py`
    + Shows the basics of using RLlib with SMARTS through :class:`~smarts.env.rllib_hiway_env.RLlibHiWayEnv`.

#. Policy gradient with population based training

    + script: :examples:`e12_rllib/ppo_pbt_example.py`
    + Combines Proximal Policy Optimization with `Population Based Training (PBT) <https://docs.ray.io/en/latest/tune/api/doc/ray.tune.schedulers.PopulationBasedTraining.html>`_ scheduling.


Recommended reads
-----------------

``RLlib`` is implemented on top of ``Ray``. ``Ray`` is a distributed computing framework specifically designed with RL in mind. There are
many docs about ``Ray`` and ``RLlib``. We recommend to read the following pages first,

- `RLlib in 60 seconds <https://docs.ray.io/en/latest/rllib/rllib-training.html>`_: Getting started with ``RLlib``.
- `Common Parameters <https://docs.ray.io/en/latest/rllib/rllib-training.html#configuring-rllib-algorithms>`_: Configuring ``RLlib`` algorithms.
- `Basic Python API <https://docs.ray.io/en/latest/rllib/rllib-training.html#using-the-python-api>`_: Basic `tune` training.
- `Logging to TensorBoard <https://docs.ray.io/en/latest/tune/tutorials/tune-output.html#how-to-log-your-tune-runs-to-tensorboard>`_: How to use TensorBoard to visualize metrics.
- `Built-in Models and Preprocessors <https://docs.ray.io/en/latest/rllib/rllib-models.html#default-behaviors>`_: Built-in preprocessor, including how to deal with different observation spaces.
- `Proximal Policy Optimization (PPO) <https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#ppo>`_: ``RLlib`` PPO implementation and PPO parameters.
- `Tune Key Concepts <https://docs.ray.io/en/latest/tune/key-concepts.html>`_: Tune key concepts.
- `RLlib Examples <https://docs.ray.io/en/latest/rllib/rllib-examples.html>`_: Get to know ``RLlib`` quickly through examples.


Resume training
---------------

With respect to ``SMARTS/examples/e12_rllib`` examples, if you want to continue an aborted experiment, you can set ``resume_training=True``. But note that ``resume_training=True`` will continue to use the same configuration as was set in the original experiment.
To make changes to a started experiment, you can edit the latest experiment file in ``./results``.

Or if you want to start a new experiment but train from an existing checkpoint, you will need to look into `How to Save and Load Trial Checkpoints <https://docs.ray.io/en/latest/tune/tutorials/tune-trial-checkpoints.html#how-to-save-and-load-trial-checkpoints>`_.
