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
from ultra.utils.curriculum.coordinator import Coordinator
from ultra.utils.curriculum.curriculum_info import CurriculumInfo
from ultra.utils.episode import episodes
from itertools import cycle

seed = 2


class CurriculumTest(unittest.TestCase):
    # Put generated files and folders in this directory.
    OUTPUT_DIRECTORY = "tests/curriculum_test/"

    def test_static_curriculum_cli(self):
        log_dir = os.path.join(CurriculumTest.OUTPUT_DIRECTORY, "logs/")
        curriculum_dir = "../tests/scenarios/static_test_curriculum"

        try:
            os.system(
                f"python ultra/train.py --curriculum-mode True --curriculum-dir {curriculum_dir} \
                --headless True --episodes 6 --max-episode-steps 2 --curriculum-scenarios-root-dir tests/scenarios \
                --eval-episodes 0 --curriculum-scenarios-save-dir {CurriculumTest.OUTPUT_DIRECTORY} --log-dir {log_dir} \
                --curriculum-build-scenarios True"
            )
        except Exception as err:
            print(err)
            self.assertTrue(False)

        if os.path.exists(log_dir):
            self.assertTrue(True)

    def test_static_curriculum_cli_evaluation(self):
        log_dir = os.path.join(CurriculumTest.OUTPUT_DIRECTORY, "logs/")
        curriculum_dir = "../tests/scenarios/static_test_curriculum"

        try:
            os.system(
                f"python ultra/train.py --curriculum-mode True --curriculum-dir {curriculum_dir} \
                --headless True --episodes 6 --max-episode-steps 2 --curriculum-scenarios-root-dir tests/scenarios \
                --eval-rate 2 --eval-episodes 2 --curriculum-scenarios-save-dir {CurriculumTest.OUTPUT_DIRECTORY} --log-dir {log_dir} \
                --curriculum-build-scenarios True"
            )
        except Exception as err:
            print(err)
            self.assertTrue(False)

        if os.path.exists(log_dir):
            self.assertTrue(True)

    def test_dynamic_curriculum_cli(self):
        log_dir = os.path.join(CurriculumTest.OUTPUT_DIRECTORY, "logs/")
        curriculum_dir = "../tests/scenarios/dynamic_test_curriculum"

        try:
            os.system(
                f"python ultra/train.py --curriculum-mode True --curriculum-dir {curriculum_dir} \
                --headless True --episodes 1 --max-episode-steps 2 --curriculum-scenarios-root-dir tests/scenarios \
                --eval-episodes 0 --curriculum-scenarios-save-dir {CurriculumTest.OUTPUT_DIRECTORY} --log-dir {log_dir}"
            )
        except Exception as err:
            print(err)
            self.assertTrue(False)

        if os.path.exists(log_dir):
            self.assertTrue(True)

    def test_dynamic_curriculum_cli_evaluation(self):
        log_dir = os.path.join(CurriculumTest.OUTPUT_DIRECTORY, "logs/")
        curriculum_dir = "../tests/scenarios/dynamic_test_curriculum"

        try:
            os.system(
                f"python ultra/train.py --curriculum-mode True --curriculum-dir {curriculum_dir} \
                --headless True --episodes 1 --max-episode-steps 2 --curriculum-scenarios-root-dir tests/scenarios \
                --eval-rate 1 --eval-episodes 2 --curriculum-scenarios-save-dir {CurriculumTest.OUTPUT_DIRECTORY} --log-dir {log_dir}"
            )
        except Exception as err:
            print(err)
            self.assertTrue(False)

        if os.path.exists(log_dir):
            self.assertTrue(True)

        if os.path.exists(log_dir):
            self.assertTrue(True)

    def test_static_train_single_agent(self):
        log_dir = os.path.join(CurriculumTest.OUTPUT_DIRECTORY, "logs")
        curriculum_dir = "../tests/scenarios/static_test_curriculum"

        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)

        seed = 2
        policy_classes = ["ultra.baselines.sac:sac-v0"]

        ray.shutdown()
        try:
            ray.init(ignore_reinit_error=True)
            train(
                scenario_info=("00-gb", "test_grade1"),
                policy_classes=policy_classes,
                num_episodes=6,
                max_episode_steps=2,
                eval_info={
                    "eval_rate": 50,
                    "eval_episodes": 2,
                },
                timestep_sec=0.1,
                headless=True,
                seed=2,
                experiment_dir=None,
                save_model_only=False,
                curriculum_mode=True,
                curriculum_metadata={
                    "curriculum_dir": curriculum_dir,
                    "curriculum_build_scenarios": True,
                    "curriculum_scenarios_root_dir": "tests/scenarios",
                    "curriculum_scenarios_save_dir": "tests/curriculum_test/",
                },
                log_dir=log_dir,
            )
            ray.shutdown()
            self.assertTrue(True)
        except ray.exceptions.WorkerCrashedError as err:
            print(err)
            self.assertTrue(False)
            ray.shutdown()

    def test_static_coordinator(self):
        log_dir = os.path.join(CurriculumTest.OUTPUT_DIRECTORY, "logs/")
        curriculum_dir = "../tests/scenarios/static_test_curriculum"
        num_episodes = 6
        etag = "sac-v0"

        agent_coordinator = Coordinator(curriculum_dir, num_episodes)

        grade_iterator = iter(
            cycle(
                [
                    ["00-curriculum", "test_grade1"],
                    ["00-curriculum", "test_grade2"],
                    ["00-curriculum", "test_grade3"],
                ]
            )
        )

        # Initial admission into curriculum. Placed in the first grade
        agent_coordinator.next_train_grade()
        self.assertEqual(next(grade_iterator), agent_coordinator.train_grade[0])

        # Once in curriculum, iterate through all other grades
        for episode in episodes(num_episodes, etag=etag, log_dir=log_dir):
            graduate = agent_coordinator.graduate(episode.index, num_episodes)
            # If agent switches to new grade
            if graduate == True:
                self.assertEqual(next(grade_iterator), agent_coordinator.train_grade[0])
            # If agent has completed all levels (no cycle through levels again)
            if agent_coordinator.check_cycle_condition(episode.index):
                print("No cycling of grades -> run completed")
                break

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(CurriculumTest.OUTPUT_DIRECTORY):
            shutil.rmtree(CurriculumTest.OUTPUT_DIRECTORY)
