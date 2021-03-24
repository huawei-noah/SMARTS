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


class TuneTest(unittest.TestCase):
    # Put generated files and folders in this directory.
    OUTPUT_DIRECTORY = "tests/tune_test/"

    def test_tune_cli(self):
        TASK = "00"
        LEVEL = "easy"
        POLICY = "ppo"
        EPISODES = 1
        MAX_EPISODE_STEPS = 1200
        HEADLESS = True
        TUNE_LOG_DIR = os.path.join(TuneTest.OUTPUT_DIRECTORY, "tune_logs/")
        TUNE_OUTPUT_DIR = os.path.join(TuneTest.OUTPUT_DIRECTORY, "tune_results/")
        CONFIG_MODULE = "ultra.baselines.ppo.ppo.tune_params"
        METRIC = "episode_reward"
        MODE = "max"

        tune_command = (
            "python ultra/tune.py "
            f"--task {TASK} --level {LEVEL} --policy {POLICY} --episodes {EPISODES} "
            f"--max-episode-steps {MAX_EPISODE_STEPS} --headless {HEADLESS} "
            f"--log-dir {TUNE_LOG_DIR} --output-dir {TUNE_OUTPUT_DIR} "
            f"--config-module {CONFIG_MODULE} --metric {METRIC} --mode {MODE}"
        )

        try:
            os.system(tune_command)
        except Exception as err:
            print(err)
            self.assertTrue(False)

        self.assertTrue(os.path.exists(TUNE_LOG_DIR))
        self.assertTrue(os.path.exists(TUNE_OUTPUT_DIR))

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(TuneTest.OUTPUT_DIRECTORY):
            shutil.rmtree(TuneTest.OUTPUT_DIRECTORY)
