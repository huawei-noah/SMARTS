# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
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
import importlib
import re
from pathlib import Path

import gym
import yaml

from benchmark import common
from benchmark.metrics import basic_handler as metrics
from benchmark.utils import format
from benchmark.wrappers import rllib as rllib_wrappers
from smarts.core.agent_interface import (
    OGM,
    RGB,
    AgentInterface,
    NeighborhoodVehicles,
    Waypoints,
)
from smarts.core.controllers import ActionSpaceType


def _get_trainer(path, name):
    """ Load trainer with given file path and trainer name """

    module = importlib.import_module(path)
    trainer = module.__getattribute__(name)
    return trainer


def _make_rllib_config(config, mode="training"):
    """Generate agent configuration. `mode` can be `train` or 'evaluation', and the
    only difference on the generated configuration is the agent info adapter.
    """

    agent_config = config["agent"]
    interface_config = config["interface"]

    """ Parse the state configuration for agent """
    state_config = agent_config["state"]

    # initialize environment wrapper if the wrapper config is not None
    wrapper_config = state_config.get("wrapper", {"name": "Simple"})
    features_config = state_config["features"]
    # only for one frame, not really an observation
    frame_space = gym.spaces.Dict(common.subscribe_features(**features_config))
    action_type = ActionSpaceType(agent_config["action"]["type"])
    env_action_space = common.ActionSpace.from_type(action_type)
    wrapper_cls = getattr(rllib_wrappers, wrapper_config["name"])

    """ Parse policy configuration """
    policy_obs_space = wrapper_cls.get_observation_space(frame_space, wrapper_config)
    policy_action_space = wrapper_cls.get_action_space(env_action_space, wrapper_config)

    observation_adapter = wrapper_cls.get_observation_adapter(
        policy_obs_space, feature_configs=features_config, wrapper_config=wrapper_config
    )
    action_adapter = wrapper_cls.get_action_adapter(
        action_type, policy_action_space, wrapper_config
    )
    # policy observation space is related to the wrapper usage
    policy_config = (
        None,
        policy_obs_space,
        policy_action_space,
        config["policy"].get(
            "config", {"custom_preprocessor": wrapper_cls.get_preprocessor()}
        ),
    )

    """ Parse agent interface configuration """
    if interface_config.get("neighborhood_vehicles"):
        interface_config["neighborhood_vehicles"] = NeighborhoodVehicles(
            **interface_config["neighborhood_vehicles"]
        )

    if interface_config.get("waypoints"):
        interface_config["waypoints"] = Waypoints(**interface_config["waypoints"])

    if interface_config.get("rgb"):
        interface_config["rgb"] = RGB(**interface_config["rgb"])

    if interface_config.get("ogm"):
        interface_config["ogm"] = OGM(**interface_config["ogm"])

    interface_config["action"] = ActionSpaceType(action_type)

    """ Pack environment configuration """
    config["run"]["config"].update({"env": wrapper_cls})
    config["env_config"] = {
        "custom_config": {
            **wrapper_config,
            "reward_adapter": wrapper_cls.get_reward_adapter(observation_adapter),
            "observation_adapter": observation_adapter,
            "action_adapter": action_adapter,
            "info_adapter": metrics.agent_info_adapter
            if mode == "evaluation"
            else None,
            "observation_space": policy_obs_space,
            "action_space": policy_action_space,
        },
    }
    config["agent"] = {"interface": AgentInterface(**interface_config)}
    config["trainer"] = _get_trainer(**config["policy"]["trainer"])
    config["policy"] = policy_config

    print(format.pretty_dict(config))

    return config


def load_config(config_file, mode="training", framework="rllib"):
    """Load algorithm configuration from yaml file.

    This function support algorithm implemented with RLlib.
    """
    yaml.SafeLoader.add_implicit_resolver(
        "tag:yaml.org,2002:float",
        re.compile(
            """^(?:
                 [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
                |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
                |\\.[0-9_]+(?:[eE][-+][0-9]+)?
                |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
                |[-+]?\\.(?:inf|Inf|INF)
                |\\.(?:nan|NaN|NAN))$""",
            re.X,
        ),
        list("-+0123456789."),
    )
    base_dir = Path(__file__).absolute().parent.parent.parent
    with open(base_dir / config_file, "r") as f:
        raw_config = yaml.safe_load(f)

    if framework == "rllib":
        return _make_rllib_config(raw_config, mode)
    else:
        raise ValueError(f"Unexpected framework {framework}, support only rllib now.")
