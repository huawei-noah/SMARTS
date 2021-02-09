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
import unittest, ray, os, sys
import gym
import json
import shutil
from smarts.core.agent import AgentSpec
from smarts.zoo.registry import make
from ultra.train import train
from ultra.baselines.sac.sac.policy import SACPolicy

AGENT_ID = "001"
seed = 2


class TrainTest(unittest.TestCase):
    # @unittest.skip
    # def test_train_cli(self):
    #     try:
    #         os.system(
    #             "python ultra/train.py --task 00 --level easy --episodes 1 --log-dir ultra/tests/logs"
    #         )
    #     except Exception as err:
    #         print(err)
    #         self.assertTrue(False)

    # def test_locate_log_directory(self):
    #     log_dir = "ultra/tests/logs"
    #     try:
    #         os.system(
    #             f"python ultra/train.py --task 00 --level easy --policy ppo --episodes 1 --log-dir {log_dir}"
    #         )
    #     except Exception as err:
    #         print(err)

    #     if os.path.exists(log_dir):
    #         self.assertTrue(True)

    # def test_train_single_agent(self):
    #     seed = 2
    #     policy_class = "ultra.baselines.sac:sac-v0"

    #     ray.init(ignore_reinit_error=True)
    #     try:
    #         ray.get(
    #             train.remote(
    #                 task=("00", "easy"),
    #                 policy_class=policy_class,
    #                 num_episodes=1,
    #                 eval_info={"eval_rate": 1000, "eval_episodes": 2,},
    #                 timestep_sec=0.1,
    #                 headless=True,
    #                 seed=2,
    #                 log_dir="ultra/tests/logs",
    #             )
    #         )
    #         self.assertTrue(True)
    #         ray.shutdown()
    #     except ray.exceptions.WorkerCrashedError as err:
    #         print(err)
    #         self.assertTrue(False)
    #         ray.shutdown()

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
        if os.path.exists("ultra/tests/logs"):
            shutil.rmtree("ultra/tests/logs")

        os.system("pkill -9 ray")
