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
import json
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
        agent_id="",
    ):
        if experiment_dir:
            print(
                f"Loading spec for {agent_id} from {experiment_dir}/agent_metadata.pkl"
            )
            with open(f"{experiment_dir}/agent_metadata.pkl", "rb") as metadata_file:
                agent_metadata = dill.load(metadata_file)
                spec = agent_metadata["agent_specs"][agent_id]

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
            base_dir = os.path.join(os.path.dirname(__file__), "../")
            pool_path = os.path.join(base_dir, "agent_pool.json")

            policy_class_name = policy_class.__name__
            agent_name = None

            with open(pool_path, "r") as f:
                data = json.load(f)
                agents = data["agents"].keys()
                for agent in agents:
                    if data["agents"][agent]["class"] == policy_class_name:
                        agent_name = data["agents"][agent]["name"]
                        break

            assert agent_name != None

            adapter = BaselineAdapter(agent_name)
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
                    policy_params=adapter.policy_params, checkpoint_dir=checkpoint_dir
                ),
                agent_builder=policy_class,
                observation_adapter=adapter.observation_adapter,
                reward_adapter=adapter.reward_adapter,
            )
        return spec
