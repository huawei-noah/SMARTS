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
import os, sys
import glob, shutil
import math

from ultra.scenarios.generate_scenarios import build_scenarios

class DynamicScenarios():
    def __init__(self, rate=None):
        self.distribution = {
            "no-traffic": 1,
            "low-density": 0,
            "mid-density": 0,
            "high-density": 0,
        }
        self.root_dir = "ultra/scenarios"
        self.save_dir = "ultra/scenarios/taskgb/"
        self.rate = rate
    
    def change_distribution(self, increment_mode=True):
        print("Old distrbution:", self.distribution)

        if increment_mode:
            for key, value in self.distribution.items():
                if key == "no-traffic":
                    self.distribution["no-traffic"] -= 0.03
                else:
                    self.distribution[key] += 0.01
        
        print("New distrbution:", self.distribution)
    
    def reset_scenario_pool(self):
        base_dir = os.path.join(self.root_dir, "taskgb/t*")
        for f in glob.glob(base_dir):
            shutil.rmtree(f)

        for key, val in self.distribution.items():
            num_scenarios = math.ceil(self.rate * val)
            print(f"Num of {key} : {num_scenarios}")
            if num_scenarios != 0:
                build_scenarios(
                    task=f"taskgb",
                    level_name=key,
                    totals={"train": num_scenarios, "test": 1},
                    root_path=self.root_dir,
                    stopwatcher_behavior=None,
                    stopwatcher_route=None,
                    save_dir=self.save_dir,
                )
        
        # os.system("ls ultra/scenarios/taskgb/")
