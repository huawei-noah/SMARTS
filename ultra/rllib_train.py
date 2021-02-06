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

# Set environment to better support Ray
os.environ["MKL_NUM_THREADS"] = "1"
import time, string
import psutil, pickle, dill
import ray, torch, argparse
import numpy as np
from ray import tune
from smarts.zoo.registry import make
from ultra.env.rllib_ultra_env import RLlibUltraEnv
from ultra.baselines.rllib_agent import RLlibAgent
from smarts.core.controllers import ActionSpaceType
from ultra.baselines.ppo.ppo.policy import PPOPolicy
from ultra.baselines.ppo.ppo.network import PPONetwork

from ray.tune.schedulers import PopulationBasedTraining
from ray.rllib.evaluation.episode import MultiAgentEpisode
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.typing import PolicyID
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
import pprint
import ray.rllib.agents.ppo as ppo
from smarts.core.controllers import ActionSpaceType
from smarts.core.agent_interface import (
    AgentInterface,
    AgentType,
    OGM,
    Waypoints,
    NeighborhoodVehicles,
)

from ultra.baselines.common.yaml_loader import load_yaml
from smarts.core.agent import AgentSpec
from ultra.baselines.adapter import BaselineAdapter
from typing import Dict

num_gpus = 1 if torch.cuda.is_available() else 0


class String(gym.Space):
    def __init__(
        self, shape=None, min_length=1, max_length=180,
    ):
        self.shape = shape
        self.min_length = min_length
        self.max_length = max_length
        self.letters = string.ascii_letters + " .,!-"

    def sample(self):
        length = random.randint(self.min_length, self.max_length)
        string = ""
        for i in range(length):
            letter = random.choice(self.letters)
            string += letter
        return string

    def contains(self, x):
        return type(x) is "str" and len(x) > self.min and len(x) < self.max

def observation_space(state_description,
    social_feature_encoder_class,
    social_feature_encoder_params,
    social_capacity,
    num_social_features
    ):
    low_dim_states_shape = sum(state_description["low_dim_states"].values())
    if social_feature_encoder_class:
        social_vehicle_shape = social_feature_encoder_class(
            **social_feature_encoder_params
        ).output_dim
    else:
        social_vehicle_shape = social_capacity * num_social_features
    print('>>>>> SOCIAL SHAPE', social_vehicle_shape)
    return gym.spaces.Dict(
    {
        # "images": gym.spaces.Box(low=0, high=1e10, shape=(1,)),
        "low_dim_states": gym.spaces.Box(low=-1e10, high=1e10, shape=(low_dim_states_shape,)),
        "social_vehicles": gym.spaces.Box(low=-1e10, high=1e10, shape=(social_capacity,num_social_features)),
    })

ACTION_SPACE = gym.spaces.Box(
    low=np.array([0.0, 0.0, -1.0]),
    high=np.array([1.0, 1.0, 1.0]),
    dtype=np.float32,
    shape=(3,),
)


class Callbacks(DefaultCallbacks):
    @staticmethod
    def on_episode_start(
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: MultiAgentEpisode,
        env_index: int,
        **kwargs,
    ):
        # episode.user_data["ego_speed"] = []
        print("episode started......")

    @staticmethod
    def on_episode_step(
        worker: RolloutWorker,
        base_env: BaseEnv,
        episode: MultiAgentEpisode,
        env_index: int,
        **kwargs,
    ):
        print('EPISODE STEP')
        single_agent_id = list(episode._agent_to_last_obs)[0]
        obs = episode.last_raw_obs_for(single_agent_id)
        # episode.user_data["ego_speed"].append(obs["speed"])
        print(obs)
        # print(N)

    @staticmethod
    def on_episode_end(
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: MultiAgentEpisode,
        env_index: int,
        **kwargs,
    ):

        mean_ego_speed = np.mean(episode.user_data["ego_speed"])
        print(
            f"ep. {episode.episode_id:<12} ended;"
            f" length={episode.length:<6}"
            f" mean_ego_speed={mean_ego_speed:.2f}"
        )
        episode.custom_metrics["mean_ego_speed"] = mean_ego_speed


def train(task, num_episodes, policy_class, eval_info, timestep_sec, headless, seed):
    torch.set_num_threads(1)
    total_step = 0
    finished = False
    num_cpus = max(1, psutil.cpu_count(logical=False) - 1)
    print(">>>>>>", num_cpus)
    ray.init()  # num_cpus=num_cpus)
    # --------------------------------------------------------
    # Initialize Agent and social_vehicle encoding method
    # -------------------------------------------------------
    AGENT_ID = "007"

    from ultra.baselines.common.social_vehicle_config import get_social_vehicle_configs
    from ultra.baselines.common.state_preprocessor import get_state_description

    social_capacity = 10
    num_social_features = 4

    social_params = dict(
        encoder_key="no_encoder",
        social_policy_hidden_units=128,
        social_polciy_init_std=0.5,
        social_capacity=social_capacity,
        num_social_features=num_social_features,
        seed=2,
    )
    social_vehicle_config = get_social_vehicle_configs(**social_params)
    social_vehicle_encoder = social_vehicle_config["encoder"]
    observation_num_lookahead = 20

    state_description = get_state_description(
        social_vehicle_config=social_params,
        observation_waypoints_lookahead=observation_num_lookahead,
        action_size=2,
    )

    state_size = sum(state_description["low_dim_states"].values())
    print(">>>>", social_vehicle_encoder)
    social_feature_encoder_class = social_vehicle_encoder[
        "social_feature_encoder_class"
    ]
    social_feature_encoder_params = social_vehicle_encoder[
        "social_feature_encoder_params"
    ]
    if social_feature_encoder_class:
        state_size += social_feature_encoder_class(
            **social_feature_encoder_params
        ).output_dim
    else:
        state_size += social_capacity * num_social_features

    from ultra.baselines.ppo.ppo.rllib_network import TorchPPOModel

    ModelCatalog.register_custom_model("ppo_model", TorchPPOModel)

    adapter = BaselineAdapter(
        is_rllib=True,
        state_description=state_description,
        social_capacity=social_capacity,
        observation_num_lookahead=observation_num_lookahead,
        social_vehicle_config=social_vehicle_config,
    )
    print("MADE ADAPTER **********")

    result_dir = "ray_results"
    result_dir = Path(result_dir).expanduser().resolve().absolute()

    # print("Done")
    # policy = trainer.get_policy()
    # model = policy.model

    # # episode = Episode()
    pbt = PopulationBasedTraining(
        time_attr="time_total_s",
        metric="episode_reward_mean",
        mode="max",
        perturbation_interval=300,
        resample_probability=0.25,
        hyperparam_mutations={
            "lr": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
            "rollout_fragment_length": lambda: 200,
            "train_batch_size": lambda: 2000,
        },
    )
    rllib_policies = {
        "default_policy": (
            None,
            observation_space(
                state_description=state_description,
                social_feature_encoder_class=social_feature_encoder_class,
                social_feature_encoder_params=social_feature_encoder_params,
                social_capacity=social_capacity,
                num_social_features=num_social_features
            ),
            ACTION_SPACE,
            {
                "model": {
                    "custom_model": "ppo_model",
                    "custom_model_config": {
                        "state_description": state_description,
                        "social_vehicle_config": social_vehicle_config,
                        "observation_num_lookahead": observation_num_lookahead,
                        "social_capacity": social_capacity,
                        "action_size": 2,
                        "state_size": state_size,
                        "init_std": 0.5,
                        "hidden_units": 512,
                        "seed": 2,
                        "social_feature_encoder_class": social_feature_encoder_class,
                        "social_feature_encoder_params": social_feature_encoder_params,
                    },
                }
            },
        )
    }
    tune_config = {
        "env": RLlibUltraEnv,
        "log_level": "WARN",
        "callbacks": Callbacks,
        "framework": "torch",
        "num_workers": 1,
        "env_config": {
            "seed": seed,
            "scenario_info": task,
            "headless": headless,
            "state_description": state_description,
            "social_capacity": social_capacity,
            "observation_num_lookahead": observation_num_lookahead,
            "social_vehicle_config": social_vehicle_config,
            "eval_mode": False,
            "ordered_scenarios": False,
            "agent_specs": {
                f"AGENT-007": AgentSpec(
                    interface=AgentInterface(
                        waypoints=Waypoints(lookahead=20),
                        neighborhood_vehicles=NeighborhoodVehicles(200),
                        action=ActionSpaceType.Continuous,
                        rgb=False,
                        max_episode_steps=5,
                        debug=True,
                    ),
                    agent_params={},
                    agent_builder=None,
                    observation_adapter=adapter.observation_adapter,
                    reward_adapter=adapter.reward_adapter,
                )
            },
        },
        "multiagent": {"policies": rllib_policies},
    }
    result_dir = "ray_results"
    result_dir = Path(result_dir).expanduser().resolve().absolute()

    analysis = tune.run(
        "PPO",
        name="exp_1",
        stop={"time_total_s": 1200},
        checkpoint_freq=1,
        checkpoint_at_end=True,
        local_dir=str(result_dir),
        resume=False,
        restore=None,
        max_failures=3,
        num_samples=1,
        export_formats=["model", "checkpoint"],
        config=tune_config,
        scheduler=pbt,
    )
    print("DOne*****")


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

    num_cpus = max(
        1, psutil.cpu_count(logical=False) - 1
    )  # remove `logical=False` to use all cpus

    policy_class = "ultra.baselines.sac:sac-v0"
    # ray_kwargs = default_ray_kwargs(num_cpus=num_cpus, num_gpus=num_gpus)
    # ray.init()  # **ray_kwargs)
    # try:
    train(
        task=(args.task, args.level),
        num_episodes=int(args.episodes),
        eval_info={
            "eval_rate": float(args.eval_rate),
            "eval_episodes": int(args.eval_episodes),
        },
        timestep_sec=float(args.timestep),
        headless=args.headless,
        policy_class=policy_class,
        seed=args.seed,
    )
    # finally:
    #     time.sleep(1)
    #     ray.shutdown()
