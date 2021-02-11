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
import numpy as np
import torch, yaml, os, inspect, dill
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


class BaselineAgentSpec(AgentSpec):
    def __init__(
        self,
        policy_class,
        action_type,
        checkpoint_dir=None,
        task=None,
        max_episode_steps=1200,
        experiment_dir=None,
    ):
        pass

    def __new__(
        self,
        policy_class,
        action_type,
        checkpoint_dir=None,
        task=None,
        max_episode_steps=1200,
        experiment_dir=None,
    ):
        if experiment_dir:
            print(f"LOADING SPEC from {experiment_dir}/spec.pkl")
            with open(f"{experiment_dir}/spec.pkl", "rb") as input:
                spec = dill.load(input)
                new_spec = AgentSpec(
                    interface=spec.interface,
                    agent_params=dict(
                        policy_params=spec.agent_params["policy_params"],
                        checkpoint_dir=checkpoint_dir,
                    ),
                    agent_builder=spec.policy_builder,
                    observation_adapter=spec.observation_adapter,
                    reward_adapter=spec.reward_adapter,
                )
                spec = new_spec
        else:
            adapter = BaselineAdapter()
            policy_dir = "/".join(inspect.getfile(policy_class).split("/")[:-1])
            policy_params = load_yaml(f"{policy_dir}/params.yaml")
            spec = AgentSpec(
                interface=AgentInterface(
                    waypoints=Waypoints(lookahead=20),
                    neighborhood_vehicles=NeighborhoodVehicles(200),
                    action=action_type,
                    rgb=False,
                    max_episode_steps=max_episode_steps,
                    debug=True,
                ),
                agent_params=dict(
                    policy_params=policy_params, checkpoint_dir=checkpoint_dir
                ),
                agent_builder=policy_class,
                observation_adapter=adapter.observation_adapter,
                reward_adapter=adapter.reward_adapter,
            )
        return spec
