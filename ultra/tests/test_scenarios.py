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

from smarts.core.utils.sumo import sumolib
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
                f"python ultra/scenarios/interface.py generate --task 00 --level easy --root-dir tests/scenarios --save-dir {save_dir}"
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
                f"python ultra/scenarios/interface.py generate --task 00-multiagent --level easy --root-dir tests/scenarios --save-dir {save_dir}"
            )
            for dirpath, dirnames, files in os.walk(save_dir):
                if "traffic" in dirpath:
                    self.assertTrue("all.rou.xml" in files)
                if "missions.pkl" in files:
                    with open(
                        os.path.join(dirpath, "missions.pkl"), "rb"
                    ) as missions_file:
                        missions = pickle.load(missions_file)
                    if "train" in dirpath:  # The train scenario.
                        self.assertTrue(len(missions) == 3)
                    elif "test" in dirpath:  # The test scenario.
                        self.assertTrue(len(missions) == 1)
        except Exception as err:
            print(err)
            self.assertTrue(False)

    def test_generate_scenarios_with_offset(self):
        save_dir = os.path.join(ScenariosTest.OUTPUT_DIRECTORY, "maps/offset_test/")
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)

        # Values for begin and end offset ranges are extracted from
        # tests/scenarios/task00/config.yaml.
        BEGIN_OFFSET_RANGE = {78, 79, 80, 81}
        END_OFFSET_RANGE = {40, 41, 42, 43, 44}

        build_scenarios(
            task="task00",
            level_name="offset_test",
            stopwatcher_behavior=None,
            stopwatcher_route=None,
            root_path="tests/scenarios",
            save_dir=save_dir,
        )
        for dirpath, dirnames, files in os.walk(save_dir):
            if "missions.pkl" in files:
                with open(os.path.join(dirpath, "missions.pkl"), "rb") as missions_file:
                    missions = pickle.load(missions_file)
                # Get the first mission's route's begin and end offset.
                begin_offset = missions[0].mission.route.begin[2]
                end_offset = missions[0].mission.route.end[2]
                self.assertTrue(begin_offset in BEGIN_OFFSET_RANGE)
                self.assertTrue(end_offset in END_OFFSET_RANGE)

    def test_generate_scenarios_with_stops(self):
        save_dir = os.path.join(ScenariosTest.OUTPUT_DIRECTORY, "maps/stops_test/")
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)

        # Values for these constants are extracted from
        # tests/scenarios/task00/config.yaml.
        STOPPED_VEHICLES_PER_SCENARIO = 3
        STOPPED_VEHICLE_POSITIONS = {80, 100, 120}
        STOPPED_VEHICLE_LANES = {0, 1}
        STOPPED_VEHICLE_LANE_IDS = {"edge-south-SN_0", "edge-south-SN_1"}

        # Values for these constants are extracted from
        # ultra/scenarios/generate_scenarios.py
        STOPPED_VEHICLE_DEPART_TIME = 0
        STOPPED_VEHICLE_SPEED = 0
        STOPPED_VEHICLE_DURATION = 1000

        build_scenarios(
            task="task00",
            level_name="stops_test",
            stopwatcher_behavior=None,
            stopwatcher_route=None,
            root_path="tests/scenarios",
            save_dir=save_dir,
        )
        for dirpath, dirnames, files in os.walk(save_dir):
            if "all.rou.xml" in files:
                total_stopped_vehicles = 0
                traffic_file_path = os.path.join(dirpath, "all.rou.xml")

                for vehicle in sumolib.output.parse(traffic_file_path, "vehicle"):
                    if vehicle.hasChild("stop"):
                        total_stopped_vehicles += 1
                        stop_information = vehicle.getChild("stop")[0]

                        self.assertTrue(
                            int(vehicle.getAttribute("depart"))
                            == STOPPED_VEHICLE_DEPART_TIME
                        )
                        self.assertTrue(
                            int(vehicle.getAttribute("departPos"))
                            in STOPPED_VEHICLE_POSITIONS
                        )
                        self.assertTrue(
                            int(vehicle.getAttribute("departSpeed"))
                            == STOPPED_VEHICLE_SPEED
                        )
                        self.assertTrue(
                            int(vehicle.getAttribute("departLane"))
                            in STOPPED_VEHICLE_LANES
                        )
                        self.assertTrue(
                            stop_information.getAttribute("lane")
                            in STOPPED_VEHICLE_LANE_IDS
                        )
                        self.assertTrue(
                            int(stop_information.getAttribute("endPos"))
                            in STOPPED_VEHICLE_POSITIONS
                        )
                        self.assertTrue(
                            int(stop_information.getAttribute("duration"))
                            == STOPPED_VEHICLE_DURATION
                        )

                self.assertTrue(total_stopped_vehicles == STOPPED_VEHICLES_PER_SCENARIO)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(ScenariosTest.OUTPUT_DIRECTORY):
            shutil.rmtree(ScenariosTest.OUTPUT_DIRECTORY)
