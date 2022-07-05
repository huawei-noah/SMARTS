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
# FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
import inspect
import json
import os

import dill
import numpy as np
import torch
import yaml

import ultra.adapters as adapters
from smarts.core.agent import AgentSpec
from smarts.core.agent_interface import AgentInterface
from ultra.baselines.common.yaml_loader import load_yaml


class BaselineAgentSpec(AgentSpec):
    def __init__(
        self,
        policy_class,
        # action_type,
        policy_params=None,
        checkpoint_dir=None,
        task=None,
        max_episode_steps=1200,
        experiment_dir=None,
    ):
        pass

    def __new__(
        self,
        policy_class,
        # action_type,
        policy_params=None,
        checkpoint_dir=None,
        # task=None,
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
                    agent_builder=spec.agent_builder,
                    observation_adapter=spec.observation_adapter,
                    reward_adapter=spec.reward_adapter,
                    info_adapter=spec.info_adapter,
                )

                spec = new_spec
        else:
            # If policy_params is None, then there must be a params.yaml file in the
            # same directory as the policy_class module.
            if not policy_params:
                policy_class_module_file = inspect.getfile(policy_class)
                policy_class_module_directory = os.path.dirname(
                    policy_class_module_file
                )
                policy_params = load_yaml(
                    os.path.join(policy_class_module_directory, "params.yaml")
                )

            action_type = adapters.type_from_string(
                string_type=policy_params["action_type"]
            )
            observation_type = adapters.type_from_string(
                string_type=policy_params["observation_type"]
            )
            reward_type = adapters.type_from_string(
                string_type=policy_params["reward_type"]
            )
            info_type = adapters.AdapterType.DefaultInfo

            adapter_interface_requirements = adapters.required_interface_from_types(
                action_type, observation_type, reward_type, info_type
            )
            action_adapter = adapters.adapter_from_type(adapter_type=action_type)
            observation_adapter = adapters.adapter_from_type(
                adapter_type=observation_type
            )
            reward_adapter = adapters.adapter_from_type(adapter_type=reward_type)
            info_adapter = adapters.adapter_from_type(adapter_type=info_type)

            spec = AgentSpec(
                interface=AgentInterface(
                    **adapter_interface_requirements,
                    max_episode_steps=max_episode_steps,
                    debug=True,
                ),
                agent_params=dict(
                    policy_params=policy_params, checkpoint_dir=checkpoint_dir
                ),
                agent_builder=policy_class,
                action_adapter=action_adapter,
                observation_adapter=observation_adapter,
                reward_adapter=reward_adapter,
                info_adapter=info_adapter,
            )

        return spec
