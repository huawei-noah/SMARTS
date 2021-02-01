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
import unittest, glob, ray, subprocess
from ultra.scenarios.generate_scenarios import *
from ultra.scenarios.analysis.scenario_analysis import ScenarioAnalysis
from ultra.scenarios.analysis.sumo_experiment import (
    edge_lane_data_function,
    vehicle_data_function,
    sumo_rerouting_routine,
)

from ultra.scenarios.analysis.base_analysis import BaseAnalysis

# TODO : Once analyzer is fixed, test cases will be updated

# class AnalysisTest(unittest.TestCase):
#     def test_interface_analyze(self):
#         try:
#             save_dir = "ultra/tests/scenarios/maps/easy/"
#             if os.path.exists(save_dir):
#                 shutil.rmtree(save_dir)
#             build_scenarios(
#                 task="task00",
#                 level_name="easy",
#                 stopwatcher_behavior="aggressive",
#                 stopwatcher_route="south-west",
#                 root_path="ultra/tests/scenarios",
#                 save_dir=save_dir,
#             )
#             if not os.path.exists("ultra/tests/scenarios/output"):
#                 os.makedirs("ultra/tests/scenarios/output")
#             scenarios = glob.glob(f"{save_dir}/map*0*")
#             os.system(
#                 "python ultra/scenarios/interface.py analyze --scenarios ultra/tests/scenarios/maps/easy/map*0* --max-steps 2000 --end-by-stopwatcher --output ultra/tests/scenarios/output"
#             )
#             for dirpath, dirnames, files in os.walk("ultra/tests/scenarios/maps/easy/"):
#                 if "traffic" in dirpath:
#                     self.assertTrue("all.rou.xml" in files)

#             self.assertTrue(os.path.exists("ultra/tests/scenarios/output/analysis.pkl"))
#         except Exception as err:
#             print(err)
#             self.assertTrue(False)

#     def test_analyze_scenario(self):
#         save_dir = "ultra/tests/scenarios/maps/easy/"
#         output_scenario = "ultra/tests/scenarios/output"
#         if os.path.exists(save_dir):
#             shutil.rmtree(save_dir)
#         if os.path.exists(output_scenario):
#             shutil.rmtree(output_scenario)

#         build_scenarios(
#             task="task00",
#             level_name="easy",
#             stopwatcher_behavior="aggressive",
#             stopwatcher_route="south-west",
#             root_path="ultra/tests/scenarios",
#             save_dir=save_dir,
#         )
#         scenarios = glob.glob(f"{save_dir}/map*0*")
#         try:
#             ray.init(ignore_reinit_error=True)
#             analyzer = ScenarioAnalysis.remote()
#             ray.wait(
#                 [
#                     analyzer.run.remote(
#                         end_by_stopwatcher=True,
#                         scenarios=scenarios,
#                         ego=None,
#                         max_episode_steps=3000,
#                         policy=None,
#                         video_rate=100,
#                         timestep_sec=0.1,
#                         custom_traci_functions=[
#                             edge_lane_data_function,
#                             vehicle_data_function,
#                         ],
#                     )
#                 ]
#             )
#             if not os.path.exists(output_scenario):
#                 os.makedirs(output_scenario)
#             ray.wait([analyzer.save_data.remote(save_dir=output_scenario)])
#             self.assertTrue(os.path.exists(f"{output_scenario}/analysis.pkl"))
#         except ray.exceptions.WorkerCrashedError as err:
#             print(err)
#             self.assertTrue(False)

#     def test_save_histogram(self):
#         try:
#             figure_name = "ultra/tests/scenarios/output"
#             if not os.path.exists(figure_name):
#                 os.makedirs(figure_name)
#             analyzer = BaseAnalysis()
#             data = [1, 2, 3, 4]

#             analyzer.save_histogram(data=data, figure_name=figure_name, title="test")
#             if not os.path.exists(figure_name):
#                 os.makedirs(figure_name)

#         except Exception as err:
#             print(err)
#             self.assertTrue(False)
