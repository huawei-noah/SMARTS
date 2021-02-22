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
from pathlib import Path
import timeit, datetime

# Set environment to better support Ray
os.environ["MKL_NUM_THREADS"] = "1"
import time
import psutil, dill, torch
import ray, torch, argparse
import numpy as np
from ray import tune
from smarts.zoo.registry import make
from ultra.env.rllib_ultra_env import RLlibUltraEnv


from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.models import ModelCatalog
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.sac as sac
import ray.rllib.agents.ddpg as ddpg
from ray.rllib.agents.sac.sac_torch_model import SACTorchModel
from ray.rllib.agents.ddpg.ddpg_torch_model import DDPGTorchModel
from smarts.core.controllers import ActionSpaceType
from smarts.core.agent_interface import (
    AgentInterface,
    OGM,
    Waypoints,
    NeighborhoodVehicles,
)
import tempfile
from ray.tune.logger import Logger, UnifiedLogger
from ultra.baselines.rllib_models.fc_network import CustomFCModel
from ultra.baselines.common.yaml_loader import load_yaml
from smarts.core.agent import AgentSpec
from ultra.baselines.adapter import BaselineAdapter
from ultra.utils.log_info import LogInfo
from ultra.utils.common import gen_experiment_name

num_gpus = 1 if torch.cuda.is_available() else 0


class Callbacks(DefaultCallbacks):
    @staticmethod
    def on_episode_start(
        worker, base_env, policies, episode, **kwargs,
    ):
        episode.user_data = LogInfo()

    @staticmethod
    def on_episode_step(
        worker, base_env, episode, **kwargs,
    ):

        single_agent_id = list(episode._agent_to_last_obs)[0]
        policy_id = episode.policy_for(single_agent_id)
        agent_reward_key = (single_agent_id, policy_id)

        info = episode.last_info_for(single_agent_id)
        reward = episode.agent_rewards[agent_reward_key]
        if info:
            episode.user_data.add(info, reward)

    @staticmethod
    def on_episode_end(
        worker, base_env, policies, episode, **kwargs,
    ):
        episode.user_data.normalize(episode.length)
        for key, val in episode.user_data.data.items():
            if not isinstance(val, (list, tuple, np.ndarray)):
                episode.custom_metrics[key] = val

        print(
            f"Episode {episode.episode_id} ended:\nlength:{episode.length},\nenv_score:{episode.custom_metrics['env_score']},\ncollision:{episode.custom_metrics['collision']}, \nreached_goal:{episode.custom_metrics['reached_goal']},\ntimeout:{episode.custom_metrics['timed_out']},\noff_road:{episode.custom_metrics['off_road']},\ndist_travelled:{episode.custom_metrics['dist_travelled']},\ngoal_dist:{episode.custom_metrics['goal_dist']}"
        )
        print("--------------------------------------------------------")


def log_creator():
    result_dir = "ray_results"
    result_dir = Path(result_dir).expanduser().resolve().absolute()
    logdir_prefix = gen_experiment_name()

    def logger_creator(config):
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        logdir = tempfile.mkdtemp(prefix=logdir_prefix, dir=result_dir)
        return UnifiedLogger(config, logdir, loggers=None)

    return logger_creator


def train(task, num_episodes, eval_info, timestep_sec, headless, seed):

    # --------------------------------------------------------
    # Initialize Agent and social_vehicle encoding method
    # -------------------------------------------------------
    AGENT_ID = "007"

    social_vehicle_params = dict(
        encoder_key="pointnet_encoder",
        social_policy_hidden_units=128,
        social_polciy_init_std=0.5,
        num_social_features=4,
        seed=seed,
        observation_num_lookahead=20,
        social_capacity=10,
    )
    adapter = BaselineAdapter(social_vehicle_params=social_vehicle_params,)

    ModelCatalog.register_custom_model("fc_model", CustomFCModel)
    # ModelCatalog.register_custom_model("fc_model", DDPGTorchModel)
    config = ddpg.DEFAULT_CONFIG.copy()

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
        # "seed":seed,
        # checking if scenarios are switching correctly
        # the interval config
        # "train_batch_size" : 200, # Number of timesteps collected for each SGD round.
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
        # "custom_eval_function": could be used for env_score?
        "env_config": {
            "seed": seed,
            "scenario_info": task,
            "headless": headless,
            "eval_mode": False,
            "ordered_scenarios": False,
            "agent_specs": agent_specs,
            "timestep_sec": timestep_sec,
        },
        "multiagent": {
            "policies": rllib_policies,
            # "policy_mapping_fn": policy_mapping
            # "replay_mode": "independent",
            # Which metric to use as the "batch size" when building a
            # MultiAgentBatch. The two supported values are:
            # env_steps: Count each time the env is "stepped" (no matter how many
            #   multi-agent actions are passed/how many multi-agent observations
            #   have been returned in the previous step).
            # agent_steps: Count each individual agent step as one step.
            # "count_steps_by": "env_steps",
        },
        # ---------------
        # "train_batch_size":1200, # remove after debugging
    }
    config.update(tune_config)
    trainer = ppo.PPOTrainer(
        env=RLlibUltraEnv, config=tune_config, logger_creator=log_creator(),
    )

    # Iteration value in trainer.py (self._iterations) is the technically the number of episodes
    for i in range(num_episodes):
        results = trainer.train()
        trainer.log_result(results)  # Evaluation will now display on Tensorboard


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
        "--episodes", help="number of training episodes", type=int, default=1000000
    )
    parser.add_argument(
        "--timestep", help="environment timestep (sec)", type=float, default=0.1
    )
    parser.add_argument(
        "--headless", help="run without envision", type=bool, default=False
    )
    parser.add_argument(
        "--eval-episodes", help="number of evaluation episodes", type=int, default=200
    )
    parser.add_argument(
        "--eval-rate",
        help="evaluation rate based on number of observations",
        type=int,
        default=10000,
    )

    parser.add_argument(
        "--seed", help="environment seed", default=2, type=int,
    )
    args = parser.parse_args()

    ray.init()
    train(
        task=(args.task, args.level),
        num_episodes=int(args.episodes),
        eval_info={
            "eval_rate": float(args.eval_rate),
            "eval_episodes": int(args.eval_episodes),
        },
        timestep_sec=float(args.timestep),
        headless=args.headless,
        seed=args.seed,
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
