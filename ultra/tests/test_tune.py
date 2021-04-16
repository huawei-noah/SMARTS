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

from ultra.tune import _AVAILABLE_TUNE_METRICS
from ultra.utils.episode import LogInfo


class TuneTest(unittest.TestCase):
    # Put generated files and folders in this directory.
    OUTPUT_DIRECTORY = "tests/tune_test/"

    def test_tune_cli(self):
        TASK = "00"
        LEVEL = "easy"
        POLICY = "ppo"
        EPISODES = 2
        MAX_EPISODE_STEPS = 50
        TIMESTEP = 0.1
        HEADLESS = True
        EVAL_EPISODES = 2
        SAVE_RATE = 1
        SEED = 2
        LOG_DIR = os.path.join(TuneTest.OUTPUT_DIRECTORY, "tune_logs/")
        CONFIG_MODULE = "tests.tune.ppo_config"
        METRIC = "episode_return"
        MODE = "max"
        SCOPE = "last"
        GRACE_PERIOD = 1
        REDUCTION_FACTOR = 2
        BRACKETS = 1
        NUM_SAMPLES = 2

        tune_command = (
            "python ultra/tune.py "
            f"--task {TASK} "
            f"--level {LEVEL} "
            f"--policy {POLICY} "
            f"--episodes {EPISODES} "
            f"--max-episode-steps {MAX_EPISODE_STEPS} "
            f"--timestep {TIMESTEP} "
            f"--headless {HEADLESS} "
            f"--eval-episodes {EVAL_EPISODES} "
            f"--save-rate {SAVE_RATE} "
            f"--seed {SEED} "
            f"--log-dir {LOG_DIR} "
            f"--config-module {CONFIG_MODULE} "
            f"--metric {METRIC} "
            f"--mode {MODE} "
            f"--scope {SCOPE} "
            f"--grace-period {GRACE_PERIOD} "
            f"--reduction-factor {REDUCTION_FACTOR} "
            f"--brackets {BRACKETS} "
            f"--num-samples {NUM_SAMPLES}"
        )

        try:
            os.system(tune_command)
        except Exception as err:
            print(err)
            self.assertTrue(False)

        # Ensure tuning was performed.
        self.assertTrue(os.path.exists(LOG_DIR))
        tune_experiment_dir = glob.glob(os.path.join(LOG_DIR, "*/"))[0]
        tune_experiment_evaluation_dir = os.path.join(
            tune_experiment_dir, "evaluation/"
        )

        # Ensure evaluation was performed.
        self.assertTrue(os.path.exists(tune_experiment_evaluation_dir))
        self.assertTrue(len(os.listdir(tune_experiment_evaluation_dir)) > 0)

        # Ensure the best parameters were saved.
        self.assertTrue(
            any(
                filename.endswith(".yaml")
                for filename in os.listdir(tune_experiment_dir)
            )
        )

    def test_tune_metrics_availability(self):
        log_info = LogInfo()
        for metric in _AVAILABLE_TUNE_METRICS:
            self.assertTrue(metric in log_info.data)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(TuneTest.OUTPUT_DIRECTORY):
            shutil.rmtree(TuneTest.OUTPUT_DIRECTORY)
