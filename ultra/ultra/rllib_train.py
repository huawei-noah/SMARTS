# MIT License
#
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
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
import os, gym
from ultra.utils.ray import default_ray_kwargs
import timeit, datetime

# Set environment to better support Ray
os.environ["MKL_NUM_THREADS"] = "1"
import time
import psutil, dill, torch, inspect
import ray, torch, argparse
import numpy as np
from ray import tune
from smarts.zoo.registry import make
from ultra.env.rllib_ultra_env import RLlibUltraEnv

from ray.rllib.models import ModelCatalog

from smarts.core.controllers import ActionSpaceType
from smarts.core.agent_interface import (
    AgentInterface,
    OGM,
    Waypoints,
    NeighborhoodVehicles,
)

from ultra.baselines.rllib.models.fc_network import CustomFCModel
from ultra.baselines.rllib.agent import RllibAgent
from ultra.baselines.common.yaml_loader import load_yaml
from smarts.core.agent import AgentSpec
from ultra.baselines.adapter import BaselineAdapter

from ultra.utils.episode import Callbacks
from ultra.utils.episode import log_creator

num_gpus = 1 if torch.cuda.is_available() else 0


def train(
    task,
    num_episodes,
    policy,
    eval_info,
    timestep_sec,
    headless,
    seed,
    training_batch_size,
    log_dir,
):

    agent_name = policy
    adapter = BaselineAdapter(agent_name)
    ModelCatalog.register_custom_model("fc_model", CustomFCModel)
    config = RllibAgent.rllib_default_config(agent_name)

    rllib_policies = {
        "default_policy": (
            None,
            adapter.observation_space,
            adapter.action_space,
            {
                "model": {
                    "custom_model": "fc_model",
                    "custom_model_config": {"adapter": adapter},
                }
            },
        )
    }
    agent_specs = {
        "AGENT-007": AgentSpec(
            interface=AgentInterface(
                waypoints=Waypoints(lookahead=20),
                neighborhood_vehicles=NeighborhoodVehicles(200),
                action=ActionSpaceType.Continuous,
                rgb=False,
                max_episode_steps=600,
                debug=True,
            ),
            agent_params={},
            agent_builder=None,
            observation_adapter=adapter.observation_adapter,
            reward_adapter=adapter.reward_adapter,
            # action_adapter=adapter.action_adapter,
        )
    }

    tune_config = {
        "env": RLlibUltraEnv,
        "log_level": "WARN",
        "callbacks": Callbacks,
        "framework": "torch",
        "num_workers": 1,
        "train_batch_size": training_batch_size,  # Debugging value
        "in_evaluation": True,
        "evaluation_num_episodes": eval_info["eval_episodes"],
        "evaluation_interval": eval_info[
            "eval_rate"
        ],  # Evaluation occurs after # of eval-intervals (episodes)
        "evaluation_config": {
            "env_config": {
                "seed": seed,
                "scenario_info": task,
                "headless": headless,
                "eval_mode": True,
                "ordered_scenarios": False,
                "agent_specs": agent_specs,
                "timestep_sec": timestep_sec,
            },
            "explore": False,
        },
        "env_config": {
            "seed": seed,
            "scenario_info": task,
            "headless": headless,
            "eval_mode": False,
            "ordered_scenarios": False,
            "agent_specs": agent_specs,
            "timestep_sec": timestep_sec,
        },
        "multiagent": {"policies": rllib_policies},
    }

    config.update(tune_config)
    agent = RllibAgent(
        agent_name=agent_name,
        env=RLlibUltraEnv,
        config=tune_config,
        logger_creator=log_creator(log_dir),
    )

    # Iteration value in trainer.py (self._iterations) is the technically the number of episodes
    for i in range(num_episodes):
        results = agent.train()
        agent.log_evaluation_metrics(
            results
        )  # Evaluation metrics will now be displayed on Tensorboard


if __name__ == "__main__":
    parser = argparse.ArgumentParser("intersection-single-agent")
    parser.add_argument(
        "--task", help="Tasks available : [0, 1, 2, 3]", type=str, default="1"
    )
    parser.add_argument(
        "--level",
        help="Tasks available : [easy, medium, hard, no-traffic]",
        type=str,
        default="easy",
    )
    parser.add_argument(
        "--policy",
        help="Policies avaliable : [ppo, ddpg, td3]",
        type=str,
        default="ppo",
    )
    parser.add_argument(
        "--episodes", help="number of training episodes", type=int, default=1000000
    )
    parser.add_argument(
        "--timestep", help="environment timestep (sec)", type=float, default=0.1
    )
    parser.add_argument(
        "--headless", help="run without envision", type=bool, default=False
    )
    parser.add_argument(
        "--eval-episodes", help="number of evaluation episodes", type=int, default=100
    )
    parser.add_argument(
        "--eval-rate",
        help="run evaluation every 'n' number of iterations",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--seed",
        help="environment seed",
        default=2,
        type=int,
    )
    parser.add_argument(
        "--training-batch-size",
        help="maximum samples for each training iteration",
        default=4000,
        type=int,
    )
    parser.add_argument(
        "--log-dir",
        help="Log directory location",
        default="logs",
        type=str,
    )

    args = parser.parse_args()

    ray.init()
    train(
        task=(args.task, args.level),
        num_episodes=int(args.episodes),
        policy=args.policy,
        eval_info={
            "eval_rate": int(args.eval_rate),
            "eval_episodes": int(args.eval_episodes),
        },
        timestep_sec=float(args.timestep),
        headless=args.headless,
        seed=args.seed,
        training_batch_size=args.training_batch_size,
        log_dir=args.log_dir,
    )


# keeping for later
# from ray.tune.schedulers import PopulationBasedTraining
# pbt = PopulationBasedTraining(
#     time_attr="time_total_s",
#     metric="episode_reward_mean",
#     mode="max",
#     perturbation_interval=300,
#     resample_probability=0.25,
#     hyperparam_mutations={
#         "lr": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
#         "rollout_fragment_length": lambda: 200,
#         "train_batch_size": lambda: 2000,
#     },
# )

# analysis = tune.run(
#     "PPO",  # "SAC"
#     name="exp_1",
#     stop={"training_iteration": 10},  # {"timesteps_total": 1200},
#     checkpoint_freq=10,
#     checkpoint_at_end=True,
#     local_dir=str(result_dir),
#     resume=False,
#     restore=None,
#     max_failures=1,
#     num_samples=1,
#     export_formats=["model", "checkpoint"],
#     config=config,
#     loggers=
#     # scheduler=pbt,
#     # "lr": tune.grid_search([1e-3,1e-4])
# )
