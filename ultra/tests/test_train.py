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
import os
import shutil
import sys
import unittest

import gym
import ray

from smarts.core.agent import AgentSpec
from smarts.zoo.registry import make
from ultra.baselines.sac.sac.policy import SACPolicy
from ultra.train import train

AGENT_ID = "001"
seed = 2


class TrainTest(unittest.TestCase):
    # Put generated files and folders in this directory.
    OUTPUT_DIRECTORY = "tests/train_test/"

    def test_train_cli(self):
        log_dir = os.path.join(TrainTest.OUTPUT_DIRECTORY, "logs/")
        try:
            os.system(
                f"python ultra/train.py --task 00 --level easy --episodes 1 --max-episode-steps 2 --log-dir {log_dir}"
            )
        except Exception as err:
            print(err)
            self.assertTrue(False)

        if os.path.exists(log_dir):
            self.assertTrue(True)

    def test_train_cli_multiagent(self):
        log_dir = os.path.join(TrainTest.OUTPUT_DIRECTORY, "logs/")
        try:
            os.system(
                f"python ultra/train.py --task 00-multiagent --level easy --episodes 1 --max-episodes-steps 2 --log-dir {log_dir} --policy dqn,bdqn,ppo"
            )
        except Exception as err:
            print(err)
            self.assertTrue(False)

        if os.path.exists(log_dir):
            self.assertTrue(True)

    def test_train_single_agent(self):
        log_dir = os.path.join(TrainTest.OUTPUT_DIRECTORY, "logs/")
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)

        seed = 2
        policy_classes = ["ultra.baselines.sac:sac-v0"]

        ray.shutdown()
        try:
            ray.init(ignore_reinit_error=True)
            ray.wait(
                [
                    train.remote(
                        scenario_info=("00", "easy"),
                        policy_classes=policy_classes,
                        num_episodes=1,
                        max_episode_steps=2,
                        eval_info={
                            "eval_rate": 1000,
                            "eval_episodes": 2,
                        },
                        timestep_sec=0.1,
                        headless=True,
                        seed=2,
                        log_dir=log_dir,
                    )
                ]
            )
            ray.shutdown()
            self.assertTrue(True)
        except ray.exceptions.WorkerCrashedError as err:
            print(err)
            self.assertTrue(False)
            ray.shutdown()

    def test_train_multiagent(self):
        log_dir = os.path.join(TrainTest.OUTPUT_DIRECTORY, "logs/")
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)

        seed = 2
        policy_classes = [
            "ultra.baslines.sac:sac-v0",
            "ultra.baselines.ppo:ppo-v0",
            "ultra.baselines.td3:td3-v0",
        ]

        ray.shutdown()
        try:
            ray.init(ignore_reinit_error=True)
            ray.wait(
                [
                    train.remote(
                        scenario_info=("00-multiagent", "easy"),
                        policy_classes=policy_classes,
                        num_episodes=1,
                        max_episode_steps=2,
                        eval_info={
                            "eval_rate": 1000,
                            "eval_episodes": 2,
                        },
                        timestep_sec=0.1,
                        headless=True,
                        seed=2,
                        log_dir=log_dir,
                    )
                ]
            )
            ray.shutdown()
            self.assertTrue(True)
        except ray.exceptions.WorkerCrashedError as err:
            print(err)
            self.assertTrue(False)
            ray.shutdown()

    def test_check_agents_from_pool(self):
        seed = 2
        policy = ""

        with open("ultra/agent_pool.json") as f:
            data = json.load(f)
            for policy in data["agents"].keys():
                policy_path = data["agents"][policy]["path"]
                policy_locator = data["agents"][policy]["locator"]
                policy_class = str(policy_path) + ":" + str(policy_locator)
                try:
                    spec = make(locator=policy_class)
                    agent = spec.build_agent()
                except ImportError as err:
                    self.assertTrue(False)

    def test_spec_is_instance_agentspec(self):
        policy_class = "ultra.baselines.sac:sac-v0"
        spec = make(locator=policy_class)
        self.assertIsInstance(spec, AgentSpec)

    def test_agent_is_instance_policy(self):
        policy_class = "ultra.baselines.sac:sac-v0"
        spec = make(locator=policy_class)
        agent = spec.build_agent()
        self.assertIsInstance(agent, SACPolicy)

    def tearDown(self):
        log_dir = os.path.join(TrainTest.OUTPUT_DIRECTORY, "logs/")
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(TrainTest.OUTPUT_DIRECTORY):
            shutil.rmtree(TrainTest.OUTPUT_DIRECTORY)
