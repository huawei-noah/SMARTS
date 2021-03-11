# RLlib support in ULTRA

## Summary

We have integrated Ray's Reinforcement Learning framework inside ULTRA to cater to users who are familiar with RLlib. We have blended ULTRA's environment and agent specifications inside RLlib's configuration to allow their policies to interact with the ULTRA environment. We currently support RLlib's PPO policy, but plan to support SAC, TD3 and DQN.

## Setup

To use RLlib with ULTRA, you must have the ultra-rl>=0.2. If you have not followed the setup guide for ULTRA, here is a [link](docs/setup.md) to it 

## Training agents

The training process is directly adopted from our own train script. More information on ULTRA's training methodology can be found [here](docs/getting_started.md). A quick rundown
of the steps is shown below

1) To generated maps used by the ULTRA scenarios, run the following command: 
  ```sh
  $ scl scenario build-all ultra/scenarios/pool
  ```
2) For building scenarios from specified task configurations, run the following command
  ```sh
  $ python ultra/scenarios/interface.py generate --task <TASK> --level <LEVEL>
  ```
3) Execute `ultra/rllib_train.py`. The following is a list of available arguments.
  - `--task`: The task number to run (default is 1).
  - `--level`: The level of the task (default is easy).
  - `--episodes`: The number of training episodes to run (default is 100).
  - `--timestep`: The environment timestep in seconds (default is 0.1).
  - `--headless`: Whether to run training without Envision (default is False).
  - `--eval-episodes`: The number of evaluation episodes (default is 200).
  - `--eval-rate`: The rate at which evaluation occurs based on the number of episodes (default is 10000).
  - `--seed`: The environment seed (default is 2).
  - `--policy`: The policy (agent) to train (default is ppo). Only PPO is supported for now.
  - `--log-dir`: The directory to put models, tensorboard data, and training results (default is logs/).
  - `--training-batch-samples` : The number of trainig samples per iteration (default is 4000).

  An example to show the how to run rllib training
  ```sh
  $ python ultra/rllib_train.py --task 1 --level easy --models logs/<timestamped_experiment_name>/models/ --episodes 5 --max-samples 200
  ```
  > This will produce another experiment directory under `logs/` containing the results of the training/testing
  
  ## View Tensorboard Results

  To view the Tensorboard results of this experiment, run the command below:
  ```sh
  $ tensorboard --logdir <absolute_path_to_ULTRA>/logs/<timestamped_experiment_name>
  ```
  > View the result in your browser with the provided link.
