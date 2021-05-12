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
from pathlib import Path

import gym

from benchmark.agents import load_config
from smarts.core.agent import AgentSpec
from smarts.core.scenario import Scenario


def gen_config(**kwargs):
    scenario_path = Path(kwargs["scenario"]).absolute()
    agent_missions_count = Scenario.discover_agent_missions_count(scenario_path)
    if agent_missions_count == 0:
        agent_ids = ["default_policy"]
    else:
        agent_ids = [f"AGENT-{i}" for i in range(agent_missions_count)]

    config = load_config(kwargs["config_file"], mode=kwargs.get("mode", "training"))
    agents = {agent_id: AgentSpec(**config["agent"]) for agent_id in agent_ids}

    config["env_config"].update(
        {
            "seed": 42,
            "scenarios": [str(scenario_path)],
            "headless": kwargs["headless"],
            "agent_specs": agents,
        }
    )

    obs_space, act_space = config["policy"][1:3]
    tune_config = config["run"]["config"]

    if kwargs["paradigm"] == "centralized":
        config["env_config"].update(
            {
                "obs_space": gym.spaces.Tuple([obs_space] * agent_missions_count),
                "act_space": gym.spaces.Tuple([act_space] * agent_missions_count),
                "groups": {"group": agent_ids},
            }
        )
        tune_config.update(config["policy"][-1])
    else:
        policies = {}
        for k in agents:
            policies[k] = config["policy"][:-1] + (
                {**config["policy"][-1], "agent_id": k},
            )
        tune_config.update(
            {
                "multiagent": {
                    "policies": policies,
                    "policy_mapping_fn": lambda agent_id: agent_id,
                }
            }
        )

    return config
