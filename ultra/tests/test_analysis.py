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
import subprocess
import unittest

import ray

from ultra.scenarios.analysis.base_analysis import BaseAnalysis
from ultra.scenarios.analysis.scenario_analysis import ScenarioAnalysis
from ultra.scenarios.analysis.sumo_experiment import (
    edge_lane_data_function,
    sumo_rerouting_routine,
    vehicle_data_function,
)
from ultra.scenarios.generate_scenarios import *


class AnalysisTest(unittest.TestCase):
    # Put generated files and folders in this directory.
    OUTPUT_DIRECTORY = "tests/analysis_test/"

    def test_interface_analyze(self):
        try:
            save_dir = os.path.join(AnalysisTest.OUTPUT_DIRECTORY, "scenarios/")
            output = os.path.join(AnalysisTest.OUTPUT_DIRECTORY, "output/")
            if os.path.exists(save_dir):
                shutil.rmtree(save_dir)
            if os.path.exists(output):
                shutil.rmtree(save_dir)
            build_scenarios(
                task="task00",
                level_name="easy",
                stopwatcher_behavior="aggressive",
                stopwatcher_route="south-west",
                root_path="tests/scenarios",
                save_dir=save_dir,
            )

            if not os.path.exists(output):
                os.makedirs(output)

            os.system(
                f"python ultra/scenarios/interface.py analyze --scenarios {save_dir} --max-steps 600 --end-by-stopwatcher --output {output}"
            )
            for dirpath, dirnames, files in os.walk(save_dir):
                if "traffic" in dirpath:
                    self.assertTrue("all.rou.xml" in files)

            self.assertTrue(os.path.exists("tests/analysis_test/output/analysis.pkl"))
        except Exception as err:
            print(err)
            self.assertTrue(False)

        if os.path.exists(save_dir):
            self.assertTrue(True)
            shutil.rmtree(save_dir)

        if os.path.exists(output):
            self.assertTrue(True)
            shutil.rmtree(output)

    def test_analyze_scenario(self):
        save_dir = os.path.join(AnalysisTest.OUTPUT_DIRECTORY, "scenarios/")
        output = os.path.join(AnalysisTest.OUTPUT_DIRECTORY, "output/")
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        if os.path.exists(output):
            shutil.rmtree(output)

        build_scenarios(
            task="task00",
            level_name="easy",
            stopwatcher_behavior="aggressive",
            stopwatcher_route="south-west",
            root_path="tests/scenarios",
            save_dir=save_dir,
        )
        scenarios = glob.glob(f"{save_dir}")
        try:
            ray.init(ignore_reinit_error=True)
            analyzer = ScenarioAnalysis.remote()
            ray.wait(
                [
                    analyzer.run.remote(
                        end_by_stopwatcher=True,
                        scenarios=scenarios,
                        ego=None,
                        max_episode_steps=800,
                        policy=None,
                        video_rate=100,
                        timestep_sec=0.1,
                        custom_traci_functions=[
                            edge_lane_data_function,
                            vehicle_data_function,
                        ],
                    )
                ]
            )
            if not os.path.exists(output):
                os.makedirs(output)
            ray.wait([analyzer.save_data.remote(save_dir=output)])
            self.assertTrue(os.path.exists(f"{output}analysis.pkl"))

            for dirpath, dirnames, files in os.walk(save_dir):
                if "traffic" in dirpath:
                    self.assertTrue("all.rou.xml" in files)

            self.assertTrue(os.path.exists("tests/analysis_test/output/analysis.pkl"))
        except ray.exceptions.WorkerCrashedError as err:
            print(err)
            self.assertTrue(False)

        if os.path.exists(save_dir):
            self.assertTrue(True)
            shutil.rmtree(save_dir)

        if os.path.exists(output):
            self.assertTrue(True)
            shutil.rmtree(output)

    def test_save_histogram(self):
        try:
            figure_name = os.path.join(AnalysisTest.OUTPUT_DIRECTORY, "histogram")
            if not os.path.exists(figure_name):
                os.makedirs(figure_name)
            analyzer = BaseAnalysis()
            data = [1, 2, 3, 4]

            analyzer.save_histogram(data=data, figure_name=figure_name, title="test")
            if not os.path.exists(figure_name):
                os.makedirs(figure_name)

        except Exception as err:
            print(err)
            self.assertTrue(False)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(AnalysisTest.OUTPUT_DIRECTORY):
            shutil.rmtree(AnalysisTest.OUTPUT_DIRECTORY)
