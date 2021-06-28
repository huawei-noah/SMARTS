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
import unittest
import os, shutil


class BaselinesTest(unittest.TestCase):
    # Put generated files and folders in this directory.
    OUTPUT_DIRECTORY = "tests/baselines_test/"

    def test_train_evaluate_baselines(self):
        """Ensure that all baselines are trainable and testable. Evaluating
        the baselines will test their save() and load() methods"""
        BASELINES = ["bdqn", "dqn", "ppo", "sac", "td3"]
        log_dir = os.path.join(BaselinesTest.OUTPUT_DIRECTORY, "logs/")
        for baseline in BASELINES:
            try:
                os.system(
                    f"python ultra/train.py --policy {baseline} --task 00 --level easy --episodes 1 --eval-episodes 2 "
                    f"--max-episode-steps 2 --log-dir {log_dir} --headless"
                )
            except Exception as err:
                print(err)
                self.assertTrue(False)

            if not os.path.exists(log_dir):
                self.assertTrue(False)
            else:
                shutil.rmtree(log_dir)
