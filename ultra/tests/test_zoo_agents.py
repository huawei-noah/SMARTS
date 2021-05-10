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
import glob
import os
import shutil
import unittest

import ray

from smarts.core.agent import Agent, AgentSpec
from smarts.core.agent_interface import (
    AgentBehavior,
    AgentInterface,
    AgentType,
    NeighborhoodVehicles,
    OGM,
)
from smarts.core.controllers import ActionSpaceType
from smarts.zoo.registry import register
from ultra.train import train


class KeepLaneAgent(Agent):
    def act(self, _):
        return "keep_lane"

    # TODO: Remove if/when smarts.core.agent.Agent has this method.
    def save(self, _):
        pass


register(
    locator="waypoints_test_agent-v0",
    entry_point=lambda: AgentSpec(
        interface=AgentInterface(waypoints=True, action=ActionSpaceType.Lane),
        agent_builder=KeepLaneAgent,
    ),
)
register(
    locator="laner_test_agent-v0",
    entry_point=lambda: AgentSpec(
        interface=AgentInterface.from_type(AgentType.Laner),
        agent_builder=KeepLaneAgent,
    ),
)
register(
    locator="behavior_test_agent-v0",
    entry_point=lambda: AgentSpec(
        interface=AgentInterface(
            waypoints=True,
            neighborhood_vehicles=True,
            agent_behavior=AgentBehavior(aggressiveness=1.0),
            action=ActionSpaceType.Lane,
        ),
        agent_builder=KeepLaneAgent,
    ),
)
register(
    locator="ogm_test_agent-v0",
    entry_point=lambda: AgentSpec(
        interface=AgentInterface(
            waypoints=True,
            neighborhood_vehicles=NeighborhoodVehicles(radius=60),
            ogm=OGM(64, 64, 0.25),
            action=ActionSpaceType.Lane,
        ),
        agent_builder=KeepLaneAgent,
    ),
)


class ZooAgentsTest(unittest.TestCase):
    # Put generated files and folders in this directory.
    OUTPUT_DIRECTORY = "tests/zoo_agents_test/"

    def test_training_zoo_agents(self):
        """Test SMARTS zoo agents in ULTRA's train script."""
        TASK = "00"
        LEVEL = "easy"
        NUM_EPISODES = 2
        MAX_EPISODE_STEPS = 10
        EVAL_RATE = 1
        EVAL_EPISODES = 1
        TIMESTEP_SEC = 0.1
        HEADLESS = True
        SEED = 2
        AGENT_LOCATORS = [
            "tests.test_zoo_agents:laner_test_agent-v0",
            "tests.test_zoo_agents:waypoints_test_agent-v0",
            "tests.test_zoo_agents:behavior_test_agent-v0",
            "tests.test_zoo_agents:ogm_test_agent-v0",
        ]

        for locator in AGENT_LOCATORS:
            log_directory = os.path.join(ZooAgentsTest.OUTPUT_DIRECTORY, locator)
            agent_infos = {
                "000": {
                    "locator": locator,
                    "spec_train_params": {},  # Expect agent to take no train params.
                    "spec_eval_params": {},  # Expect agent to take no eval params.
                },
            }

            ray.init(ignore_reinit_error=True)
            train(
                scenario_info=(TASK, LEVEL),
                num_episodes=NUM_EPISODES,
                agent_infos=agent_infos,
                max_episode_steps=MAX_EPISODE_STEPS,
                eval_info={
                    "eval_rate": EVAL_RATE,
                    "eval_episodes": EVAL_EPISODES,
                },
                timestep_sec=TIMESTEP_SEC,
                headless=HEADLESS,
                seed=SEED,
                log_dir=log_directory,
            )

            experiment_directory = glob.glob(os.path.join(log_directory, "*/"))[0]
            pkls_directory = os.path.join(experiment_directory, "pkls/")
            agent_metadata_file = os.path.join(
                experiment_directory, "agent_metadata.pkl"
            )
            failure_message = f"Failed on '{locator}'."

            self.assertTrue(os.path.exists(log_directory), msg=failure_message)
            self.assertTrue(os.path.exists(experiment_directory), msg=failure_message)
            self.assertTrue(os.path.exists(pkls_directory), msg=failure_message)
            self.assertTrue(len(os.listdir(pkls_directory)) > 0, msg=failure_message)
            self.assertTrue(os.path.isfile(agent_metadata_file), msg=failure_message)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(ZooAgentsTest.OUTPUT_DIRECTORY):
            shutil.rmtree(ZooAgentsTest.OUTPUT_DIRECTORY)
