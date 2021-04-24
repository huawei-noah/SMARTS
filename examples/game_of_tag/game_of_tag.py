"""Let's play tag!

A predator-prey multi-agent example built on top of RLlib to facilitate further
developments on multi-agent support for HiWay (including design, performance,
research, and scaling).

The predator and prey use separate policies. A predator "catches" its prey when
it collides into the other vehicle. There can be multiple predators and
multiple prey in a map. Social vehicles act as obstacles where both the
predator and prey must avoid them.
"""
import argparse
import os
import random
import multiprocessing
import ray


import numpy as np
from gym.spaces import Tuple
from typing import List
from ray import tune
from ray.rllib.utils import try_import_tf
from ray.rllib.models import ModelCatalog
from ray.tune import Stopper, register_env, grid_search
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.tune.schedulers import PopulationBasedTraining

# from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.agents.qmix.qmix import QMixTrainer
from pathlib import Path

from smarts.env.rllib_hiway_env import RLlibHiWayEnv
from smarts.core.agent import AgentSpec, Agent
from smarts.core.controllers import ActionSpaceType
from smarts.core.agent_interface import AgentInterface, AgentType, DoneCriteria
from smarts.core.utils.file import copy_tree


from examples.game_of_tag.tag_adapters import *
from examples.game_of_tag.model import CustomFCModel


# Add custom metrics to your tensorboard using these callbacks
# see: https://ray.readthedocs.io/en/latest/rllib-training.html#callbacks-and-custom-metrics
def on_episode_start(info):
    episode = info["episode"]
    print("episode {} started".format(episode.episode_id))


def on_episode_step(info):
    episode = info["episode"]
    single_agent_id = list(episode._agent_to_last_obs)[0]
    obs = episode.last_raw_obs_for(single_agent_id)


def on_episode_end(info):
    episode = info["episode"]


def explore(config):
    # ensure we collect enough timesteps to do sgd
    if config["train_batch_size"] < config["sgd_minibatch_size"] * 2:
        config["train_batch_size"] = config["sgd_minibatch_size"] * 2
    # ensure we run at least one sgd iter
    if config["num_sgd_iter"] < 1:
        config["num_sgd_iter"] = 1
    return config


def policy_mapper(agent_id):
    if agent_id in PREDATOR_IDS:
        return "predator_policy"
    elif agent_id in PREY_IDS:
        return "prey_policy"


class TimeStopper(Stopper):
    def __init__(self):
        self._start = time.time()
        self._deadline = 48 * 60 * 60  # 9 hours

    def __call__(self, trial_id, result):
        return False

    def stop_all(self):
        return time.time() - self._start > self._deadline


tf = try_import_tf()


# NUM_SOCIAL_VEHICLES = 10 ######### Why we have to define how many social vehicles?

# # to make this network smaller, neural network accept bits?
# class TrainingModel(FullyConnectedNetwork):
#     NAME = "FullyConnectedNetwork"


ModelCatalog.register_custom_model("CustomFCModel", CustomFCModel)

rllib_agents = {}
# add custom done criteria - maybe not
# map offset difference between sumo-gui and envision

shared_interface = AgentInterface(
    max_episode_steps=1500,
    neighborhood_vehicles=True,
    waypoints=True,
    action=ActionSpaceType.LaneWithContinuousSpeed,
)
shared_interface.done_criteria = DoneCriteria(
    off_route=False,
    wrong_way=False,
    collision=True,
)  # off_road=False? Try to still have off_road event but off_road=False in done creteria
# shared_interface.neighborhood_vehicles = NeighborhoodVehicles(radius=50) # To-do have different radius for prey vs predator

# predator_neighborhood_vehicles=NeighborhoodVehicles(radius=30)
# print(f'model location: {os.path.join(os.path.dirname(os.path.realpath(__file__)), "model")}')
for agent_id in PREDATOR_IDS:
    rllib_agents[agent_id] = {
        "agent_spec": AgentSpec(
            interface=shared_interface,
            agent_builder=lambda: TagModelAgent(  ## maybe fine since it might understand which mode it is in. Try 2 models at first
                os.path.join(os.path.dirname(os.path.realpath(__file__)), "model"),
                OBSERVATION_SPACE,
            ),
            observation_adapter=observation_adapter,
            reward_adapter=predator_reward_adapter,
            action_adapter=action_adapter,
        ),
        "observation_space": OBSERVATION_SPACE,
        "action_space": ACTION_SPACE,
    }

for agent_id in PREY_IDS:
    rllib_agents[agent_id] = {
        "agent_spec": AgentSpec(
            interface=shared_interface,
            agent_builder=lambda: TagModelAgent(
                os.path.join(
                    os.path.dirname(os.path.realpath(__file__)), "model"
                ),  # assume model exists
                OBSERVATION_SPACE,
            ),
            observation_adapter=observation_adapter,
            reward_adapter=prey_reward_adapter,
            action_adapter=action_adapter,
        ),
        "observation_space": OBSERVATION_SPACE,
        "action_space": ACTION_SPACE,
    }


def get_agent_id(id):
    if id in PREY_IDS:
        return 0
    elif id in PREDATOR_IDS:
        return 1
    else:
        raise Exception("Error: Wrong id")


def env_creator(config):
    grouping = {
        "PREY1": PREY_IDS,
        "PRED1": PREDATOR_IDS,
    }
    env = RLlibHiWayEnv(config)
    return env.with_agent_groups(
        grouping, obs_space=OBSERVATION_SPACE, act_space=ACTION_SPACE
    )


register_env("tag_env", env_creator)


def build_tune_config(scenario, headless):
    rllib_policies = {
        policy_mapper(agent_id): (
            None,
            rllib_agent["observation_space"],
            rllib_agent["action_space"],
            # TrainingModel is a FullyConnectedNetwork, which is the default in rllib
            {
                # "model": {"custom_model": 'CustomFCModel'},
                "agent_id": agent_id,
            },
        )
        for agent_id, rllib_agent in rllib_agents.items()
    }

    # tune_config = {
    #     "env": "tag_env",
    #     #"framework": "tf2", # Can't export model
    #     "framework": "torch",
    #     "log_level": "WARN",
    #     "num_workers": 3,
    #     "explore": True,
    #     # 'sample_batch_size': 200,  # XXX: 200
    #     # 'train_batch_size': 4000,
    #     # 'sgd_minibatch_size': 128,
    #     # 'num_sgd_iter': 30,
    #     "horizon": 10000,
    #     "env_config": {
    #         "seed": 42,
    #         "sim_name": "game_of_tag_works?",
    #         "scenarios": [os.path.abspath(scenario)],
    #         "headless": headless,
    #         "agent_specs": {
    #             agent_id: rllib_agent["agent_spec"]
    #             for agent_id, rllib_agent in rllib_agents.items()
    #         },
    #     },
    #     "multiagent": {
    #         "policies": rllib_policies,
    #         "policies_to_train": ["predator_policy", "prey_policy"],
    #         #"policy_mapping_fn":  lambda agent_id: f"{agent_id}_policy",
    #         "policy_mapping_fn": policy_mapper,
    #     },
    #     "callbacks": {
    #         "on_episode_start": on_episode_start,
    #         "on_episode_step": on_episode_step,
    #         "on_episode_end": on_episode_end,
    #     },
    # }
    # return tune_config

    config = {
        "env": "tag_env",
        "rollout_fragment_length": 4,
        "train_batch_size": 1500,
        "exploration_config": {
            "epsilon_timesteps": 5000,
            "final_epsilon": 0.05,
        },
        "num_workers": 2,
        "mixer": "qmix",
        "env_config": {
            "seed": 42,
            "sim_name": "game_of_tag_works?",
            "scenarios": [os.path.abspath(scenario)],
            "headless": headless,
            "agent_specs": {
                agent_id: rllib_agent["agent_spec"]
                for agent_id, rllib_agent in rllib_agents.items()
            },
        },
        "multiagent": {
            "policies": rllib_policies,
            "policies_to_train": ["predator_policy", "prey_policy"],
            # "policy_mapping_fn":  lambda agent_id: f"{agent_id}_policy",
            "policy_mapping_fn": policy_mapper,
        },
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "framework": "torch",
    }
    return config


def main(args):
    # pbt = PopulationBasedTraining(
    #     time_attr="time_total_s",
    #     metric="episode_reward_mean",
    #     mode="max",
    #     perturbation_interval=300,
    #     resample_probability=0.25,
    #     # Specifies the mutations of these hyperparams
    #     hyperparam_mutations={
    #         "lambda": lambda: random.uniform(0.9, 1.0),
    #         "clip_param": lambda: random.uniform(0.01, 0.5),
    #         "kl_coeff": lambda: 0.3,
    #         "lr": [1e-3], # test to see if it works
    #         #"num_sgd_iter": lambda: random.randint(1, 30),
    #         "sgd_minibatch_size": lambda: 128, #random.randint(128, 16384),
    #         # "train_batch_size": lambda: random.randint(2000, 160000),
    #         "train_batch_size": lambda: 4000,
    #         'num_sgd_iter': lambda: 30,
    #     },
    #     custom_explore_fn=explore,
    # )
    local_dir = os.path.expanduser(args.result_dir)

    tune_config = build_tune_config(args.scenario, args.headless)

    analysis = tune.run(
        QMixTrainer,
        name="lets_play_tag",
        # stop={'time_total_s': 10 * 60 },#60 * 60 * 12},  # 12 hours
        stop=TimeStopper(),
        # XXX: Every X iterations perform a _ray actor_ checkpoint (this is
        #      different than _exporting_ a TF/PT checkpoint).
        checkpoint_freq=5,
        checkpoint_at_end=True,
        # XXX: Beware, resuming after changing tune params will not pick up
        #      the new arguments as they are stored alongside the checkpoint.
        resume=args.resume_training,
        # restore="/home/kyber/ray_results/lets_play_tag/PPO_RLlibHiWayEnv_77a55_00000_0_2021-04-06_12-59-36/checkpoint_133/checkpoint-133",
        local_dir=local_dir,
        reuse_actors=True,
        max_failures=0,
        export_formats=["model", "checkpoint"],
        config=tune_config,
        # scheduler=pbt,
    )

    # print(analysis.dataframe().head())

    # best_logdir = Path(analysis.get_best_logdir("episode_reward_max", mode="max"))
    # model_path = best_logdir / "model"
    # print(f'best reward directory model {model_path}')

    # save_model_path = str(Path(__file__).expanduser().resolve().parent / "model")
    # copy_tree(str(model_path), save_model_path, overwrite=True)
    # print(f"Wrote model to: {save_model_path}")

    # #supposed to output a model
    # checkpoint_path = '/home/kyber/ray_results/lets_play_tag/PPO_RLlibHiWayEnv_66f3c_00000_0_2021-04-20_17-21-42/checkpoint_6/checkpoint-6'
    # ray.init(num_cpus=2)
    # training_agent = PPOTrainer(env=RLlibHiWayEnv,config=tune_config)
    # training_agent.restore(checkpoint_path)
    # prefix = "model.ckpt"
    # model_dir = os.path.join(os.getcwd(), "prey_model")
    # print(f'model path: {model_dir}')
    # #training_agent.export_policy_checkpoint('model', filename_prefix=prefix)
    # training_agent.export_policy_model(model_dir)
    # print('works')


if __name__ == "__main__":
    parser = argparse.ArgumentParser("rllib-example")
    parser.add_argument(
        "scenario",
        type=str,
        help="Scenario to run (see scenarios/ for some samples you can use)",
    )
    parser.add_argument(
        "--headless", help="run simulation in headless mode", action="store_true"
    )
    parser.add_argument(
        "--resume_training",
        default=False,
        action="store_true",
        help="Resume the last trained example",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default="~/ray_results",
        help="Directory containing results (and checkpointing)",
    )
    args = parser.parse_args()
    main(args)
