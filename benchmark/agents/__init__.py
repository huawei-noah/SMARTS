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
import yaml
import gym
import importlib
import pprint
import re

from pathlib import Path

from smarts.core.agent_interface import OGM, RGB, Waypoints, NeighborhoodVehicles
from smarts.core.controllers import ActionSpaceType
from smarts.env.rllib_hiway_env import RLlibHiWayEnv

from benchmark.wrappers import rllib as rllib_wrappers
from benchmark.metrics import basic_metrics as metrics

from . import common


def _get_trainer(path, name):
    module = importlib.import_module(path)
    trainer = module.__getattribute__(name)
    return trainer


def _get_policy_observation_space(wrapper, observation_space, config):
    if wrapper == rllib_wrappers.FrameStack:
        return common.get_stacked_space(
            observation_space, config["parameters"]["num_stack"]
        )
    else:
        return observation_space


def _make_rllib_config(config, mode="train"):
    """ Generate agent configuration. `mode` can be `train` or 'evaluation', and the
    only difference on the generated configuration is the agent info adapter.
    """

    agent = config["agent"]
    state_config = agent["state"]

    # initialize environment wrapper if the wrapper config is not None
    wrapper_config = state_config["wrapper"]
    wrapper = (
        getattr(rllib_wrappers, wrapper_config["name"]) if wrapper_config else None
    )

    features = state_config["features"]
    # only for one frame, not really an observation
    frame_space = gym.spaces.Dict(common.subscribe_features(**features))

    action_type = agent["action"]["type"]
    action_space = common.ActionSpace.from_type(action_type)

    # policy observation space is related to the wrapper usage
    policy_obs_space = _get_policy_observation_space(
        wrapper, frame_space, wrapper_config
    )
    policy_config = (
        None,
        policy_obs_space,
        action_space,
        config["policy"].get("config", {}),
    )

    interface = config["interface"]
    if interface.get("neighborhood_vehicles"):
        interface["neighborhood_vehicles"] = NeighborhoodVehicles(
            **interface["neighborhood_vehicles"]
        )

    if interface.get("waypoints"):
        interface["waypoints"] = Waypoints(**interface["waypoints"])

    if interface.get("rgb"):
        interface["rgb"] = RGB(**interface["rgb"])

    if interface.get("ogm"):
        interface["ogm"] = OGM(**interface["ogm"])

    interface["action"] = ActionSpaceType(action_type)

    adapter_type = "vanilla" if wrapper == rllib_wrappers.FrameStack else "single_frame"

    agent_config = dict(
        action_adapter=common.ActionAdapter.from_type(action_type),
        info_adapter=metrics.agent_info_adapter
        if mode == "evaluate"
        else common.default_info_adapter,
    )

    adapter_type = (
        "stack_frame" if wrapper == rllib_wrappers.FrameStack else "single_frame"
    )
    observation_adapter = common.get_observation_adapter(
        frame_space, adapter_type, wrapper=wrapper, feature_configs=features
    )

    env_config = dict()
    wrapper_config["parameters"] = {
        "observation_adapter": observation_adapter,
        "reward_adapter": common.get_reward_adapter(observation_adapter, adapter_type),
        "base_env_cls": RLlibHiWayEnv,
    }
    env_config.update(**wrapper_config["parameters"])
    config["run"]["config"].update({"env": wrapper})

    config["env_config"] = env_config
    config["agent"] = agent_config
    config["interface"] = interface
    config["trainer"] = _get_trainer(**config["policy"]["trainer"])
    config["policy"] = policy_config

    pprint.pprint(config)

    return config


def load_config(config_file, mode="train", framework="rllib"):
    yaml.SafeLoader.add_implicit_resolver(
        u"tag:yaml.org,2002:float",
        re.compile(
            u"""^(?:
                 [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
                |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
                |\\.[0-9_]+(?:[eE][-+][0-9]+)?
                |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
                |[-+]?\\.(?:inf|Inf|INF)
                |\\.(?:nan|NaN|NAN))$""",
            re.X,
        ),
        list(u"-+0123456789."),
    )
    base_dir = Path(__file__).absolute().parent.parent
    with open(base_dir / config_file, "r") as f:
        raw_config = yaml.safe_load(f)

    if framework == "rllib":
        return _make_rllib_config(raw_config, mode)
    else:
        raise NotImplementedError
