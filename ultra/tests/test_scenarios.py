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
import pickle
import shutil
import unittest

from smarts.core.utils.sumo import sumolib
from smarts.sstudio.types import MapZone, PositionalZone
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
            root_path="tests/scenarios/",
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
            root_path="tests/scenarios/",
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

    def test_interface_generate_with_shuffled_missions(self):
        TASK = "00-multiagent"
        LEVEL = "shuffle_test"
        ROOT_DIR = "tests/scenarios"
        SAVE_DIR = os.path.join(
            ScenariosTest.OUTPUT_DIRECTORY, "maps/shuffle_test-with-shuffle/"
        )

        if os.path.exists(SAVE_DIR):
            shutil.rmtree(SAVE_DIR)

        generate_command = (
            "python ultra/scenarios/interface.py generate "
            f"--task {TASK} "
            f"--level {LEVEL} "
            f"--root-dir {ROOT_DIR} "
            f"--save-dir {SAVE_DIR}"
        )

        # Test generation without the --no-mission-shuffle flag.
        try:
            os.system(generate_command)
        except Exception as error:
            print(error)
            self.assertTrue(False)

        first_missions = None
        num_scenario_missions_that_differ = 0

        for dirpath, _, files in os.walk(SAVE_DIR):
            # Testing scenarios only contain one mission, therefore there is no need to
            # test if they are shuffled. Only test the training scenarios.
            if "missions.pkl" in files and "train" in dirpath:
                with open(os.path.join(dirpath, "missions.pkl"), "rb") as missions_file:
                    missions = pickle.load(missions_file)
                if not first_missions:
                    first_missions = missions
                # Check to see if this current scenario's missions are in a different
                # order as the first scenario's missions. Check equality of missions by
                # checking that their start and end lanes are the same.
                self.assertTrue(len(missions) == len(first_missions))
                for mission, first_mission in zip(missions, first_missions):
                    if (
                        mission.mission.route.begin[0]
                        != first_mission.mission.route.begin[0]
                        or mission.mission.route.end[0]
                        != first_mission.mission.route.end[0]
                    ):
                        num_scenario_missions_that_differ += 1

        # Ensure that at least one scenario's mission order differs from the first
        # scenario's mission order.
        self.assertTrue(num_scenario_missions_that_differ > 0)

    def test_interface_generate_without_shuffled_missions(self):
        TASK = "00-multiagent"
        LEVEL = "shuffle_test"
        ROOT_DIR = "tests/scenarios"
        SAVE_DIR = os.path.join(
            ScenariosTest.OUTPUT_DIRECTORY, "maps/shuffle_test-without-shuffle/"
        )

        if os.path.exists(SAVE_DIR):
            shutil.rmtree(SAVE_DIR)

        generate_command = (
            "python ultra/scenarios/interface.py generate "
            "--no-mission-shuffle "
            f"--task {TASK} "
            f"--level {LEVEL} "
            f"--root-dir {ROOT_DIR} "
            f"--save-dir {SAVE_DIR}"
        )

        # Test generation with the --no-mission-shuffle flag.
        try:
            os.system(generate_command)
        except Exception as error:
            print(error)
            self.assertTrue(False)

        first_missions = None

        for dirpath, _, files in os.walk(SAVE_DIR):
            # Testing scenarios only contain one mission, therefore there is no need to
            # test if they are shuffled. Only test the training scenarios.
            if "missions.pkl" in files and "train" in dirpath:
                with open(os.path.join(dirpath, "missions.pkl"), "rb") as missions_file:
                    missions = pickle.load(missions_file)
                if not first_missions:
                    first_missions = missions
                # Ensure that this current scenario's missions are in the same order as
                # the first scenario's missions. Check equality of missions by checking
                # that their start and end lanes are the same.
                self.assertTrue(len(missions) == len(first_missions))
                for mission, first_mission in zip(missions, first_missions):
                    self.assertEqual(
                        mission.mission.route.begin[0],
                        first_mission.mission.route.begin[0],
                    )
                    self.assertEqual(
                        mission.mission.route.end[0], first_mission.mission.route.end[0]
                    )

    def test_generate_scenarios_with_offset(self):
        save_dir = os.path.join(ScenariosTest.OUTPUT_DIRECTORY, "maps/offset_test/")
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)

        # Values for begin and end offset ranges are extracted from
        # tests/scenarios/task00/config.yaml under the offset test level.
        BEGIN_OFFSET_RANGE = {78, 79, 80, 81}
        END_OFFSET_RANGE = {40, 41, 42, 43, 44}

        build_scenarios(
            task="task00",
            level_name="offset_test",
            stopwatcher_behavior=None,
            stopwatcher_route=None,
            root_path="tests/scenarios/",
            save_dir=save_dir,
        )

        for scenario_directory in glob.glob(os.path.join(save_dir, "*/")):
            missions_file_path = os.path.join(scenario_directory, "missions.pkl")

            self.assertTrue(os.path.isfile(missions_file_path))

            with open(missions_file_path, "rb") as missions_file:
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
        # tests/scenarios/task00/config.yaml under the stops test level.
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
            root_path="tests/scenarios/",
            save_dir=save_dir,
        )

        for scenario_directory in glob.glob(os.path.join(save_dir, "*/")):
            traffic_file_path = os.path.join(scenario_directory, "traffic/all.rou.xml")

            self.assertTrue(os.path.isfile(traffic_file_path))

            total_stopped_vehicles = 0

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
                        int(vehicle.getAttribute("departLane")) in STOPPED_VEHICLE_LANES
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

    def test_generate_scenarios_with_bubbles(self):
        save_dir = os.path.join(ScenariosTest.OUTPUT_DIRECTORY, "maps/bubbles_test/")
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)

        # Values for these constants are extracted from
        # tests/scenarios/task00/config.yaml under the bubbles test level.
        INTERSECTION_BUBBLE_SIZE = (40, 50)
        INTERSECTION_BUBBLE_ACTOR_NAME = "intersection-actor"
        INTERSECTION_BUBBLE_AGENT_LOCATOR = "intersection_test_agent:test-agent-v0"
        INTERSECTION_BUBBLE_AGENT_PARAMS = {}
        LANE_BUBBLE_LANE = "edge-south-SN"
        LANE_BUBBLE_LANE_INDEX = 0
        LANE_BUBBLE_OFFSET = 30
        LANE_BUBBLE_LENGTH = 50
        LANE_BUBBLE_NUM_LANES = 2
        LANE_BUBBLE_ACTOR_NAME = "lane-actor"
        LANE_BUBBLE_AGENT_LOCATOR = "lane_test_agent:test-agent-v0"
        LANE_BUBBLE_AGENT_PARAMS = {"speed": 30}

        build_scenarios(
            task="task00",
            level_name="bubbles_test",
            stopwatcher_behavior=None,
            stopwatcher_route=None,
            root_path="tests/scenarios/",
            save_dir=save_dir,
        )

        for scenario_directory in glob.glob(os.path.join(save_dir, "*/")):
            bubbles_file_path = os.path.join(scenario_directory, "bubbles.pkl")

            self.assertTrue(os.path.isfile(bubbles_file_path))

            with open(bubbles_file_path, "rb") as bubbles_file:
                bubbles = pickle.load(bubbles_file)

            for bubble in bubbles:
                if isinstance(bubble.zone, PositionalZone):
                    self.assertTrue(bubble.zone.size == INTERSECTION_BUBBLE_SIZE)
                    self.assertTrue(bubble.actor.name == INTERSECTION_BUBBLE_ACTOR_NAME)
                    self.assertTrue(
                        bubble.actor.agent_locator == INTERSECTION_BUBBLE_AGENT_LOCATOR
                    )
                    self.assertTrue(
                        bubble.actor.policy_kwargs == INTERSECTION_BUBBLE_AGENT_PARAMS
                    )
                elif isinstance(bubble.zone, MapZone):
                    self.assertTrue(bubble.zone.start[0] == LANE_BUBBLE_LANE)
                    self.assertTrue(bubble.zone.start[1] == LANE_BUBBLE_LANE_INDEX)
                    self.assertTrue(bubble.zone.start[2] == LANE_BUBBLE_OFFSET)
                    self.assertTrue(bubble.zone.length == LANE_BUBBLE_LENGTH)
                    self.assertTrue(bubble.zone.n_lanes == LANE_BUBBLE_NUM_LANES)
                    self.assertTrue(bubble.actor.name == LANE_BUBBLE_ACTOR_NAME)
                    self.assertTrue(
                        bubble.actor.agent_locator == LANE_BUBBLE_AGENT_LOCATOR
                    )
                    self.assertTrue(
                        bubble.actor.policy_kwargs == LANE_BUBBLE_AGENT_PARAMS
                    )
                else:
                    self.assertTrue(False)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(ScenariosTest.OUTPUT_DIRECTORY):
            shutil.rmtree(ScenariosTest.OUTPUT_DIRECTORY)
