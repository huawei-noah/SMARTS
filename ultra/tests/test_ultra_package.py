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
import os
import shutil
import unittest

import ray

from ultra.scenarios.generate_scenarios import build_scenarios
from ultra.train import train


class UltraPackageTest(unittest.TestCase):
    # Put generated files and folders in this directory.
    OUTPUT_DIRECTORY = "tests/ultra_package_test/"

    def test_simple_train_run(self):
        save_dir = "tests/task/eval_test/"
        log_dir = os.path.join(UltraPackageTest.OUTPUT_DIRECTORY, "tests/logs")

        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)

        save_dir = "tests/task/eval_test/eval"

        try:
            build_scenarios(
                task="task00",
                level_name="eval_test",
                stopwatcher_behavior=None,
                stopwatcher_route=None,
                root_path="tests/scenarios",
                save_dir=save_dir,
            )
        except:
            self.assertTrue(False)

        policy_class = "ultra.baselines.sac:sac-v0"

        ray.shutdown()
        try:
            ray.init(ignore_reinit_error=True)
            ray.wait(
                [
                    train.remote(
                        scenario_info=("00", "eval_test"),
                        policy_classes=[policy_class],
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
            ray.shutdown()
            self.assertTrue(False)

        save_dir = "tests/task/eval_test/"

        if len(os.listdir(save_dir)) != 0:
            self.assertTrue(True)
            shutil.rmtree(save_dir)
        else:
            self.assertTrue(False)

        if os.path.exists(log_dir):
            self.assertTrue(True)
            shutil.rmtree(log_dir)
        else:
            self.assertTrue(False)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(UltraPackageTest.OUTPUT_DIRECTORY):
            shutil.rmtree(UltraPackageTest.OUTPUT_DIRECTORY)
        if os.path.exists("tests/task/eval_test/"):
            shutil.rmtree("tests/task/eval_test/")
