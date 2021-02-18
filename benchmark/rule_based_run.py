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
import argparse
import os
from pathlib import Path

from smarts.core.agent import Agent, AgentSpec
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.scenario import Scenario
from smarts.env.hiway_env import HiWayEnv


class RuleBasedAgent(Agent):
    def act(self, obs):
        return "keep_lane"


def parse_args():
    parser = argparse.ArgumentParser("Rule based runner")
    parser.add_argument(
        "scenario",
        type=str,
        help="Scenario name",
    )

    return parser.parse_args()


def main(scenario):
    scenario_path = Path(scenario).absolute()
    agent_mission_count = Scenario.discover_agent_missions_count(scenario_path)

    assert agent_mission_count > 0, "agent mission count should larger than 0"

    agent_ids = [f"AGENT-{i}" for i in range(agent_mission_count)]

    agent_specs = {
        agent_id: AgentSpec(
            interface=AgentInterface.from_type(AgentType.Laner, max_episode_steps=None),
            agent_builder=RuleBasedAgent,
        )
        for agent_id in agent_ids
    }

    agents = {aid: agent_spec.build_agent() for aid, agent_spec in agent_specs.items()}

    env = HiWayEnv(scenarios=[scenario_path], agent_specs=agent_specs)

    while True:
        observations = env.reset()
        done = False
        while not done:
            agent_ids = list(observations.keys())
            actions = {aid: agents[aid].act(observations[aid]) for aid in agent_ids}
            observations, _, dones, _ = env.step(actions)
            done = dones["__all__"]


if __name__ == "__main__":
    args = parse_args()
    main(scenario=args.scenario)
