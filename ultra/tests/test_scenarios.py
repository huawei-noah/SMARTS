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
import pickle
import shutil
import unittest

from ultra.scenarios.generate_scenarios import *


class ScenariosTest(unittest.TestCase):
    # Put generated files and folders in this directory.
    OUTPUT_DIRECTORY = "tests/scenarios_test/"

    def test_interface_generate(self):
        try:
            save_dir = os.path.join(ScenariosTest.OUTPUT_DIRECTORY, "maps/easy/")
            if os.path.exists(save_dir):
                shutil.rmtree(save_dir)
            os.system(
                f"python ultra/scenarios/interface.py generate --task 00 --level easy --root-dir tests/scenarios --save-dir {save_dir}map"
            )
            for dirpath, dirnames, files in os.walk(save_dir):
                if "traffic" in dirpath:
                    self.assertTrue("all.rou.xml" in files)
                if "missions.pkl" in files:
                    with open(
                        os.path.join(dirpath, "missions.pkl"), "rb"
                    ) as missions_file:
                        missions = pickle.load(missions_file)
                    self.assertTrue(len(missions) == 1)
        except Exception as err:
            print(err)
            self.assertTrue(False)

    def test_generate_scenario(self):
        save_dir = os.path.join(ScenariosTest.OUTPUT_DIRECTORY, "maps/easy/")
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)

        build_scenarios(
            task="task00",
            level_name="easy",
            stopwatcher_behavior="aggressive",
            stopwatcher_route="south-west",
            root_path="tests/scenarios",
            save_dir=save_dir,
        )
        for dirpath, dirnames, files in os.walk(save_dir):
            if "traffic" in dirpath:
                self.assertTrue("all.rou.xml" in files)
            if "missions.pkl" in files:
                with open(os.path.join(dirpath, "missions.pkl"), "rb") as missions_file:
                    missions = pickle.load(missions_file)
                self.assertTrue(len(missions) == 1)

    def test_generate_no_traffic(self):
        save_dir = os.path.join(ScenariosTest.OUTPUT_DIRECTORY, "maps/no-traffic/")
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        build_scenarios(
            task="task00",
            level_name="no-traffic",
            stopwatcher_behavior="aggressive",
            stopwatcher_route="south-west",
            root_path="tests/scenarios",
            save_dir=save_dir,
        )
        for dirpath, dirnames, files in os.walk(save_dir):
            if "traffic" in dirpath:
                self.assertTrue("all.rou.xml" not in files)
            if "missions.pkl" in files:
                with open(os.path.join(dirpath, "missions.pkl"), "rb") as missions_file:
                    missions = pickle.load(missions_file)
                self.assertTrue(len(missions) == 1)

    def test_interface_generate_multiagent(self):
        try:
            save_dir = os.path.join(
                ScenariosTest.OUTPUT_DIRECTORY, "maps/easy-multiagent/"
            )
            if os.path.exists(save_dir):
                shutil.rmtree(save_dir)
            os.system(
                f"python ultra/scenarios/interface.py generate --task 00-multiagent --level easy --root-dir tests/scenarios --save-dir {save_dir}map"
            )
            for dirpath, dirnames, files in os.walk(save_dir):
                if "traffic" in dirpath:
                    self.assertTrue("all.rou.xml" in files)
                if "missions.pkl" in files:
                    with open(
                        os.path.join(dirpath, "missions.pkl"), "rb"
                    ) as missions_file:
                        missions = pickle.load(missions_file)
                    if "0" in dirpath:  # The train scenario.
                        self.assertTrue(len(missions) == 3)
                    elif "1" in dirpath:  # The test scenario.
                        self.assertTrue(len(missions) == 1)
        except Exception as err:
            print(err)
            self.assertTrue(False)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(ScenariosTest.OUTPUT_DIRECTORY):
            shutil.rmtree(ScenariosTest.OUTPUT_DIRECTORY)
