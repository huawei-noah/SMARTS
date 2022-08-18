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

import gym
import numpy as np
import ray
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.rllib.utils import try_import_tf
from ray.tune.schedulers import PopulationBasedTraining
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.agents.ppo import PPOTrainer

from examples.game_of_tag.game_of_tag import shared_interface, build_tune_config
from examples.game_of_tag.model import CustomFCModel
from examples.game_of_tag.tag_adapters import (
    OBSERVATION_SPACE,
    PREDATOR_IDS,
    PREY_IDS,
    observation_adapter,
    predator_reward_adapter,
    prey_reward_adapter,
)

from smarts.env.rllib_hiway_env import RLlibHiWayEnv
from smarts.core.agent import AgentSpec, Agent
from smarts.core.agent_interface import AgentInterface, AgentType, DoneCriteria
from smarts.core.utils.episodes import episodes
from smarts.core.controllers import ActionSpaceType

tf = try_import_tf()[1]

# must use >3 cpus since training used 3 workers
ray.init(num_cpus=4)


ModelCatalog.register_custom_model("CustomFCModel", CustomFCModel)


def action_adapter(model_action):
    """Take in the action calculated by the model, and transform it to something that
    SMARTS can understand.

    The model returns a batched action (since it received a batched input). That is, the
    action consists of actions for however many observations were passed to it in the
    batch of observations it was given. We only gave it a batch of 1 observation in the
    act(...) method of TagModelAgent.

    The model outputs an action in the form of:
        (
            (
                array([...]),  # The speed.
                array([...]),  # The lane change.
            ),
            [],
            {
                '...': array([...]),
                '...': array([[...]]),
                '...': array([...]),
                '...': array([...])
            }
        )

    The action we care about is the first element of this tuple, get it with
    model_action[0], so that speed = array([...]) and laneChange = array([...]). Convert
    these arrays to scalars to index into speeds or subtract from it.
    """
    speed, laneChange = model_action[0]
    speeds = [0, 3, 6, 9]
    adapted_action = [speeds[speed.item()], laneChange.item() - 1]
    return adapted_action


class TagModelAgent(Agent):
    def __init__(self, checkpoint_path, scenario, headless, policy_name):
        assert os.path.isfile(checkpoint_path)
        tune_config = build_tune_config(scenario, headless=headless, sumo_headless=True)
        self.agent = PPOTrainer(env=RLlibHiWayEnv, config=tune_config)
        self.agent.restore(checkpoint_path)
        self._policy_name = policy_name
        self._prep = ModelCatalog.get_preprocessor_for_space(OBSERVATION_SPACE)

    def act(self, observations):
        """Receive an observation from the environment, and compute the agent's action.

        The observation is a dictionary of an observation for a single agent. However,
        the model expects a batched observation, that is, a list of observations. To fix
        this, expand the dimensions of the observation from (n,) to (1, n) so that the
        observation fits into the model's expected input size.
        """
        obs = self._prep.transform(observations)
        obs = np.expand_dims(obs, 0)
        action = self.agent.get_policy(self._policy_name).compute_actions(obs)
        return action
