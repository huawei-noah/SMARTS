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
from typing import List
from ray import tune
from ray.rllib.utils import try_import_tf
from ray.rllib.models import ModelCatalog
from ray.tune import Stopper
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.tune.schedulers import PopulationBasedTraining
from ray.rllib.agents.ppo import PPOTrainer
from pathlib import Path

from smarts.env.rllib_hiway_env import RLlibHiWayEnv
from smarts.core.agent import AgentSpec, Agent
from smarts.core.controllers import ActionSpaceType
from smarts.core.agent_interface import (
    AgentInterface,
    AgentType,
    DoneCriteria,
    AgentsAliveDoneCriteria,
    AgentsListAlive,
)
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

PREDATOR_POLICY = "predator_policy"
PREY_POLICY = "prey_policy"

def policy_mapper(agent_id):
    if agent_id in PREDATOR_IDS:
        return PREDATOR_POLICY
    elif agent_id in PREY_IDS:
        return PREY_POLICY

class TimeStopper(Stopper):
    def __init__(self):
        self._start = time.time()
        # Currently will see obvious tag behaviour in 6 hours
        self._deadline = 48 * 60 * 60  # train for 48 hours

    def __call__(self, trial_id, result):
        return False

    def stop_all(self):
        return time.time() - self._start > self._deadline


tf = try_import_tf()

ModelCatalog.register_custom_model("CustomFCModel", CustomFCModel)

rllib_agents = {}

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
    agents_alive=AgentsAliveDoneCriteria(
        agent_lists_alive=[
            AgentsListAlive(agents_list=PREY_IDS, minimum_agents_alive_in_list=1),
            AgentsListAlive(agents_list=PREDATOR_IDS, minimum_agents_alive_in_list=1),
        ]
    ),
)

for agent_id in PREDATOR_IDS:
    rllib_agents[agent_id] = {
        "agent_spec": AgentSpec(
            interface=shared_interface,
            agent_builder=lambda: TagModelAgent(
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
                os.path.join(os.path.dirname(os.path.realpath(__file__)), "model"),
                OBSERVATION_SPACE,
            ),
            observation_adapter=observation_adapter,
            reward_adapter=prey_reward_adapter,
            action_adapter=action_adapter,
        ),
        "observation_space": OBSERVATION_SPACE,
        "action_space": ACTION_SPACE,
    }


def build_tune_config(scenario, headless=True, sumo_headless=False):
    rllib_policies = {
        policy_mapper(agent_id): (
            None,
            rllib_agent["observation_space"],
            rllib_agent["action_space"],
            {"model": {"custom_model": "CustomFCModel"}},
        )
        for agent_id, rllib_agent in rllib_agents.items()
    }

    tune_config = {
        "env": RLlibHiWayEnv,
        "framework": "torch",
        "log_level": "WARN",
        "num_workers": 3,
        "explore": True,
        "horizon": 10000,
        "env_config": {
            "seed": 42,
            "sim_name": "game_of_tag_works?",
            "scenarios": [os.path.abspath(scenario)],
            "headless": headless,
            "sumo_headless": sumo_headless,
            "agent_specs": {
                agent_id: rllib_agent["agent_spec"]
                for agent_id, rllib_agent in rllib_agents.items()
            },
        },
        "multiagent": {
            "policies": rllib_policies,
            "policies_to_train": [PREDATOR_POLICY, PREY_POLICY],
            "policy_mapping_fn": policy_mapper,
        },
        "callbacks": {
            "on_episode_start": on_episode_start,
            "on_episode_step": on_episode_step,
            "on_episode_end": on_episode_end,
        },
    }
    return tune_config
