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
from ultra.utils.coordinator import coordinator
from ultra.utils.episode import episodes
from itertools import cycle

seed = 2


class GBTrainTest(unittest.TestCase):
    # Put generated files and folders in this directory.
    OUTPUT_DIRECTORY = "tests/gb_train_test/"

    def test_gb_train_cli(self):
        log_dir = os.path.join(GBTrainTest.OUTPUT_DIRECTORY, "logs/")
        curriculum_dir = "../../tests/scenarios/grade_based_test_curriculum"
        save_dir = "tests/gb_train_test/"

        try:
            os.system(
                f"python ultra/train.py --gb-mode True --gb-curriculum-dir {curriculum_dir} --task 00 --level easy \
                --headless True --episodes 6 --max-episode-steps 2 --gb-scenarios-root-dir tests/scenarios \
                --gb-scenarios-save-dir {save_dir} --log-dir {log_dir} --gb-build-scenarios True"
            )
        except Exception as err:
            print(err)
            self.assertTrue(False)

        if os.path.exists(log_dir):
            self.assertTrue(True)

    def test_gb_train_cli_with_evaluation(self):
        log_dir = os.path.join(GBTrainTest.OUTPUT_DIRECTORY, "logs/")
        curriculum_dir = "../../tests/scenarios/grade_based_test_curriculum"
        save_dir = "tests/gb_train_test/"

        try:
            os.system(
                f"python ultra/train.py --gb-mode True --gb-curriculum-dir {curriculum_dir} --task 00 --level easy \
                --headless True --episodes 6 --max-episode-steps 2 --eval-rate 2 --gb-scenarios-root-dir tests/scenarios \
                --eval-episodes 1 --gb-scenarios-save-dir {save_dir} --log-dir {log_dir} --gb-build-scenarios True"
            )
        except Exception as err:
            print(err)
            self.assertTrue(False)

        if os.path.exists(log_dir):
            self.assertTrue(True)

    def test_gb_train_single_agent(self):
        log_dir = os.path.join(GBTrainTest.OUTPUT_DIRECTORY, "logs/")
        save_dir = "tests/scenarios/"
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
                        scenario_info=("00-gb", "test_grade1"),
                        policy_classes=policy_classes,
                        num_episodes=1,
                        max_episode_steps=2,
                        eval_info={
                            "eval_rate": 50,
                            "eval_episodes": 2,
                        },
                        timestep_sec=0.1,
                        headless=True,
                        seed=2,
                        grade_mode=True,
                        gb_info={
                            "gb_curriculum_dir": "../tests/scenarios/grade_based_test_curriculum",
                            "gb_build_scenarios": True,
                            "gb_scenarios_root_dir": True,
                            "gb_scenarios_save_dir": "../",
                        },
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

    def test_coordinator(self):
        log_dir = os.path.join(GBTrainTest.OUTPUT_DIRECTORY, "logs/")
        gb_curriculum_dir = "../../tests/scenarios/grade_based_test_curriculum"
        num_episodes = 8
        etag = "sac-v0"

        agent_coordinator = coordinator(gb_curriculum_dir)

        grade_iterator = iter(
            cycle(
                [
                    ["00-gb", "test_grade1"],
                    ["00-gb", "test_grade2"],
                    ["00-gb", "test_grade3"],
                ]
            )
        )

        for episode in episodes(num_episodes, etag=etag, log_dir=log_dir):
            switch_grade = agent_coordinator.graduate(episode.index, num_episodes)
            # If agent switches to new grade
            if switch_grade[0] == True:
                agent_coordinator.display()
                self.assertEqual(next(grade_iterator), agent_coordinator.get_grade()[0])

            # If agent has completed all levels (no cycle through levels again)
            if switch_grade[1] == True:
                finished = True
                self.assertTrue(finished)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(GBTrainTest.OUTPUT_DIRECTORY):
            shutil.rmtree(GBTrainTest.OUTPUT_DIRECTORY)
