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
import glob

import gym
import ray

from smarts.core.agent import AgentSpec
from smarts.zoo.registry import make
from ultra.baselines.sac.sac.policy import SACPolicy
from ultra.train import train, load_agents
from ultra.utils.common import combine_replay_buffer_dataset_with_episodes_results

AGENT_ID = "001"
seed = 2


class TrainTest(unittest.TestCase):
    # Put generated files and folders in this directory.
    OUTPUT_DIRECTORY = "tests/train_test/"

    @classmethod
    def setUpClass(cls):
        """Generate single agent model"""
        single_agent_model_log_dir = os.path.join(
            TrainTest.OUTPUT_DIRECTORY, "single_agent_model_logs/"
        )
        os.system(
            f"python ultra/train.py --task 00 --level easy --episodes 3 --eval-rate 2 --max-episode-steps 2 --log-dir {single_agent_model_log_dir} --eval-episodes 1 --headless"
        )

        """Generate multi agent models"""
        multi_agent_model_log_dir = os.path.join(
            TrainTest.OUTPUT_DIRECTORY, "multi_agent_model_logs/"
        )
        os.system(
            f"python ultra/train.py --task 00-multiagent --level easy --policy sac,sac,sac --episodes 3 --eval-rate 2 --max-episode-steps 2 --log-dir {multi_agent_model_log_dir} --eval-episodes 1 --headless"
        )

        """Run experiments to generate latest_replay_buffer.pkl and results.pkl"""
        combine_rb_episode_log_dir = os.path.join(
            TrainTest.OUTPUT_DIRECTORY, "combine_rb_episode_logs/"
        )
        os.system(
            f"python ultra/train.py --task 00 --level easy --policy sac --max-steps 100 --max-episode-steps 50 --log-dir {combine_rb_episode_log_dir} --eval-episodes 0 --headless"
        )

    def test_a_folders(self):
        single_agent_model_log_dir = os.path.join(
            TrainTest.OUTPUT_DIRECTORY, "single_agent_model_logs/"
        )
        if not os.path.exists(single_agent_model_log_dir):
            self.assertTrue(False)

        multi_agent_model_log_dir = os.path.join(
            TrainTest.OUTPUT_DIRECTORY, "multi_agent_model_logs/"
        )
        if not os.path.exists(multi_agent_model_log_dir):
            self.assertTrue(False)

        combine_rb_episode_log_dir = os.path.join(
            TrainTest.OUTPUT_DIRECTORY, "combine_rb_episode_logs/"
        )
        if not os.path.exists(combine_rb_episode_log_dir):
            self.assertTrue(False)

    def test_train_cli(self):
        log_dir = os.path.join(TrainTest.OUTPUT_DIRECTORY, "logs/")
        try:
            os.system(
                f"python ultra/train.py --task 00 --level easy --episodes 1 --eval-episodes 0 --max-episode-steps 2 --log-dir {log_dir}"
            )
        except Exception as err:
            print(err)
            self.assertTrue(False)

        if not os.path.exists(log_dir):
            self.assertTrue(False)

    def test_train_cli_multiagent(self):
        log_dir = os.path.join(TrainTest.OUTPUT_DIRECTORY, "logs/")
        try:
            os.system(
                f"python ultra/train.py --task 00-multiagent --level easy --episodes 1 --eval-episodes 0 --max-episode-steps 2 --log-dir {log_dir} --policy dqn,bdqn,ppo"
            )
        except Exception as err:
            print(err)
            self.assertTrue(False)

        if not os.path.exists(log_dir):
            self.assertTrue(False)

    def test_train_single_agent(self):
        log_dir = os.path.join(TrainTest.OUTPUT_DIRECTORY, "logs/")
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)

        seed = 2
        policy_classes = ["ultra.baselines.sac:sac-v0"]

        ray.shutdown()
        try:
            ray.init(ignore_reinit_error=True)
            train(
                scenario_info=("00", "easy"),
                policy_classes=policy_classes,
                num_episodes=1,
                max_episode_steps=2,
                max_steps=5,
                eval_info={
                    "eval_rate": 1000,
                    "eval_episodes": 2,
                },
                timestep_sec=0.1,
                headless=True,
                seed=2,
                log_dir=log_dir,
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
            "ultra.baselines.sac:sac-v0",
            "ultra.baselines.ppo:ppo-v0",
            "ultra.baselines.td3:td3-v0",
        ]

        ray.shutdown()
        try:
            ray.init(ignore_reinit_error=True)
            train(
                scenario_info=("00-multiagent", "easy"),
                policy_classes=policy_classes,
                num_episodes=1,
                max_episode_steps=2,
                max_steps=5,
                eval_info={
                    "eval_rate": 1000,
                    "eval_episodes": 2,
                },
                timestep_sec=0.1,
                headless=True,
                seed=2,
                log_dir=log_dir,
            )
            ray.shutdown()
            self.assertTrue(True)
        except ray.exceptions.WorkerCrashedError as err:
            print(err)
            self.assertTrue(False)
            ray.shutdown()

    def test_single_agent_train_model(self):
        """Further train the trained model"""
        log_dir = os.path.join(TrainTest.OUTPUT_DIRECTORY, "logs/")
        experiment_dir = glob.glob(
            os.path.join(TrainTest.OUTPUT_DIRECTORY, "single_agent_model_logs/*")
        )[0]

        try:
            os.system(
                f"python ultra/train.py --task 00 --level easy --episodes 1 --eval-episodes 0 --max-episode-steps 2 --experiment-dir {experiment_dir} --log-dir {log_dir} --headless"
            )
        except Exception as err:
            print(err)
            self.assertTrue(False)

        if not os.path.exists(log_dir):
            self.assertTrue(False)

        shutil.rmtree(log_dir)

    def test_multi_agent_train_model(self):
        """Further train the trained model"""
        log_dir = os.path.join(TrainTest.OUTPUT_DIRECTORY, "logs/")
        experiment_dir = glob.glob(
            os.path.join(TrainTest.OUTPUT_DIRECTORY, "multi_agent_model_logs/*")
        )[0]

        try:
            os.system(
                f"python ultra/train.py --task 00-multiagent --level easy --policy sac,sac,sac --episodes 1 --eval-episodes 0 --max-episode-steps 2 --experiment-dir {experiment_dir} --log-dir {log_dir} --headless"
            )
        except Exception as err:
            print(err)
            self.assertTrue(False)

        if not os.path.exists(log_dir):
            self.assertTrue(False)

        shutil.rmtree(log_dir)

    def test_load_single_agent(self):
        experiment_dir = glob.glob(
            os.path.join(TrainTest.OUTPUT_DIRECTORY, "single_agent_model_logs/*")
        )[0]
        agent_ids, agent_classes, agent_specs, agents = load_agents(experiment_dir)

        self.assertEqual(agent_ids[0], "000")
        self.assertEqual(agent_classes[agent_ids[0]], "ultra.baselines.sac:sac-v0")
        self.assertIsInstance(agent_specs[agent_ids[0]], AgentSpec)
        self.assertIsInstance(agents[agent_ids[0]], SACPolicy)
        self.assertGreater(len(agents[agent_ids[0]].memory), 0)

    def test_load_multi_agent(self):
        experiment_dir = glob.glob(
            os.path.join(TrainTest.OUTPUT_DIRECTORY, "multi_agent_model_logs/*")
        )[0]
        agent_ids, agent_classes, agent_specs, agents = load_agents(experiment_dir)

        test_agent_ids = ["000", "001", "002"]

        for index in range(len(agent_ids)):
            self.assertEqual(agent_ids[index], test_agent_ids[index])
            self.assertEqual(
                agent_classes[agent_ids[index]], "ultra.baselines.sac:sac-v0"
            )
            self.assertIsInstance(agent_specs[agent_ids[index]], AgentSpec)
            self.assertIsInstance(agents[agent_ids[index]], SACPolicy)
            self.assertGreater(len(agents[agent_ids[index]].memory), 0)

    def test_combine_replay_buffer_dataset_with_episodes_results(self):
        experiment_dir = glob.glob(
            os.path.join(TrainTest.OUTPUT_DIRECTORY, "combine_rb_episode_logs/*")
        )[0]
        agent_id = "000"
        active_tag = "Train"
        max_steps = 100

        episodes_data = combine_replay_buffer_dataset_with_episodes_results(
            experiment_dir, agent_id, active_tag
        )

        # Check if the length of a replay_buffer_dataset at a given episode
        # is equal to episode_length (or num_steps) - 1. The number of transitions
        # is always 1 less than the number of steps per episode
        for _, episode_data in episodes_data.items():
            self.assertEqual(
                len(episode_data["replay_buffer_dataset"]),
                episode_data["episode_results"]["episode_length"] - 1,
            )

        # The total number of transitions should equate to the max steps for the
        # entire experiment
        total_transitions = 0
        for _, episode_data in episodes_data.items():
            total_transitions += len(episode_data["replay_buffer_dataset"])
        self.assertEqual(total_transitions, max_steps)

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
