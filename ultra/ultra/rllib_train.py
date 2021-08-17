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
import os
import ultra.adapters as adapters
from ultra.baselines.common.social_vehicle_config import get_social_vehicle_configs
from ultra.utils.ray import default_ray_kwargs

# Set environment to better support Ray
os.environ["MKL_NUM_THREADS"] = "1"
import torch
import ray, torch, argparse
from ultra.env.rllib_ultra_env import RLlibUltraEnv
from ray.rllib.models import ModelCatalog

from smarts.core.controllers import ActionSpaceType
from smarts.core.agent_interface import (
    AgentInterface,
    Waypoints,
    NeighborhoodVehicles,
)

from ultra.baselines.rllib.models.fc_network import CustomFCModel
from ultra.baselines.rllib.agent import RllibAgent
from ultra.baselines.common.yaml_loader import load_yaml
from smarts.core.agent import AgentSpec

from ultra.utils.episode import Callbacks
from ultra.utils.episode import log_creator

num_gpus = 1 if torch.cuda.is_available() else 0


def train(
    task,
    num_episodes,
    max_episode_steps,
    rollout_fragment_length,
    policy,
    eval_info,
    timestep_sec,
    headless,
    seed,
    train_batch_size,
    sgd_minibatch_size,
    log_dir,
):
    agent_name = policy
    policy_params = load_yaml(f"ultra/baselines/{agent_name}/{agent_name}/params.yaml")

    action_type = adapters.type_from_string(policy_params["action_type"])
    observation_type = adapters.type_from_string(policy_params["observation_type"])
    reward_type = adapters.type_from_string(policy_params["reward_type"])

    if action_type != adapters.AdapterType.DefaultActionContinuous:
        raise Exception(
            f"RLlib training only supports the "
            f"{adapters.AdapterType.DefaultActionContinuous} action type."
        )
    if observation_type != adapters.AdapterType.DefaultObservationVector:
        # NOTE: The SMARTS observations adaptation that is done in ULTRA's Gym
        #       environment is not done in ULTRA's RLlib environment. If other
        #       observation adapters are used, they may raise an Exception.
        raise Exception(
            f"RLlib training only supports the "
            f"{adapters.AdapterType.DefaultObservationVector} observation type."
        )

    action_space = adapters.space_from_type(adapter_type=action_type)
    observation_space = adapters.space_from_type(adapter_type=observation_type)

    action_adapter = adapters.adapter_from_type(adapter_type=action_type)
    info_adapter = adapters.adapter_from_type(
        adapter_type=adapters.AdapterType.DefaultInfo
    )
    observation_adapter = adapters.adapter_from_type(adapter_type=observation_type)
    reward_adapter = adapters.adapter_from_type(adapter_type=reward_type)

    params_seed = policy_params["seed"]
    encoder_key = policy_params["social_vehicles"]["encoder_key"]
    num_social_features = observation_space["social_vehicles"].shape[1]
    social_capacity = observation_space["social_vehicles"].shape[0]
    social_policy_hidden_units = int(
        policy_params["social_vehicles"].get("social_policy_hidden_units", 0)
    )
    social_policy_init_std = int(
        policy_params["social_vehicles"].get("social_policy_init_std", 0)
    )
    social_vehicle_config = get_social_vehicle_configs(
        encoder_key=encoder_key,
        num_social_features=num_social_features,
        social_capacity=social_capacity,
        seed=params_seed,
        social_policy_hidden_units=social_policy_hidden_units,
        social_policy_init_std=social_policy_init_std,
    )

    ModelCatalog.register_custom_model("fc_model", CustomFCModel)
    config = RllibAgent.rllib_default_config(agent_name)

    rllib_policies = {
        "default_policy": (
            None,
            observation_space,
            action_space,
            {
                "model": {
                    "custom_model": "fc_model",
                    "custom_model_config": {
                        "social_vehicle_config": social_vehicle_config
                    },
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
                max_episode_steps=max_episode_steps,
                debug=True,
            ),
            agent_params={},
            agent_builder=None,
            action_adapter=action_adapter,
            info_adapter=info_adapter,
            observation_adapter=observation_adapter,
            reward_adapter=reward_adapter,
        )
    }

    tune_config = {
        "env": RLlibUltraEnv,
        "log_level": "WARN",
        "callbacks": Callbacks,
        "framework": "torch",
        "num_workers": 1,
        "train_batch_size": train_batch_size,
        "sgd_minibatch_size": sgd_minibatch_size,
        "rollout_fragment_length": rollout_fragment_length,
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
        "--max-episode-steps",
        help="Maximum number of steps per episode",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--rollout-fragment-length",
        help="Number of timesteps to use in batch",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--timestep", help="environment timestep (sec)", type=float, default=0.1
    )
    parser.add_argument(
        "--headless",
        help="Run without envision",
        action="store_true",
    )
    parser.add_argument(
        "--eval-episodes", help="number of evaluation episodes", type=int, default=100
    )
    parser.add_argument(
        "--eval-rate",
        help="The number of training episodes to wait before running the evaluation",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--seed",
        help="environment seed",
        default=2,
        type=int,
    )
    parser.add_argument(
        "--train-batch-size",
        help="maximum samples for each training iteration",
        default=4000,
        type=int,
    )
    parser.add_argument(
        "--sgd-minibatch-size",
        help="maximum samples for each training iteration",
        default=128,
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
        max_episode_steps=int(args.max_episode_steps),
        rollout_fragment_length=int(args.rollout_fragment_length),
        policy=args.policy,
        eval_info={
            "eval_rate": int(args.eval_rate),
            "eval_episodes": int(args.eval_episodes),
        },
        timestep_sec=float(args.timestep),
        headless=args.headless,
        seed=args.seed,
        train_batch_size=args.train_batch_size,
        sgd_minibatch_size=args.sgd_minibatch_size,
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
