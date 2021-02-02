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
import unittest, shutil, os
from ultra.scenarios.generate_scenarios import *


class ScenariosTest(unittest.TestCase):
    def test_interface_generate(self):
        try:
            save_dir = "ultra/tests/scenarios/maps/easy/"
            if os.path.exists(save_dir):
                shutil.rmtree(save_dir)
            os.system(
                "python ultra/scenarios/interface.py generate --task 00 --level easy --root-dir ultra/tests/scenarios --save-dir ultra/tests/scenarios/maps/easy/map"
            )
            for dirpath, dirnames, files in os.walk(save_dir):
                if "traffic" in dirpath:
                    self.assertTrue("all.rou.xml" in files)
        except Exception as err:
            print(err)
            self.assertTrue(False)

    def test_generate_scenario(self):
        save_dir = "ultra/tests/scenarios/maps/easy/"
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)

        build_scenarios(
            task="task00",
            level_name="easy",
            stopwatcher_behavior="aggressive",
            stopwatcher_route="south-west",
            root_path="ultra/tests/scenarios",
            save_dir=save_dir,
        )
        for dirpath, dirnames, files in os.walk(save_dir):
            if "traffic" in dirpath:
                self.assertTrue("all.rou.xml" in files)

    def test_generate_no_traffic(self):
        save_dir = "ultra/tests/scenarios/maps/no-traffic/"
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        build_scenarios(
            task="task00",
            level_name="no-traffic",
            stopwatcher_behavior="aggressive",
            stopwatcher_route="south-west",
            root_path="ultra/tests/scenarios",
            save_dir=save_dir,
        )
        for dirpath, dirnames, files in os.walk(save_dir):
            if "traffic" in dirpath:
                self.assertTrue("all.rou.xml" not in files)
