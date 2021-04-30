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


class DynamicScenarios:
    def __init__(self, root_dir, save_dir, rate=None):
        """Initial setup

        Args:
            root_path (str): path to tasks' directory
            save_dir (str): path to save the scenarios
            rate (int): sampling value which determines when to
                        add new scenarios
        """
        self.distribution = {
            "no-traffic": 1,
            "low-density": 0,
            "mid-density": 0,
            "high-density": 0,
        }
        self.root_dir = root_dir
        self.save_dir = save_dir
        self.rate = rate

    def change_distribution(self, increment_mode=True):
        """Changes distribution of different types of density.
        This is an hand-drawn function to test adding scenarios.
        A more robust algorithm or a teacher agent will be
        responsible in deciding which type of density to use next
        """
        # print("Old distrbution:", self.distribution)
        for key, value in self.distribution.items():
            if key == "no-traffic":
                self.distribution["no-traffic"] -= 0.03
            else:
                self.distribution[key] += 0.01
        # print("New distrbution:", self.distribution)

    def reset_scenario_pool(self, tasks, num_episodes=None):
        """Cleans up previous pool of train scenarios. A pool is defined to be a
           place where scenarios are loaded into every sample. For example,
           if the sample rate is 5, then every 5 episodes older scenarios are
           removed from the pool and new scenarios are added. The size of pool
           is dependent on the sampling rate.

        Args:
            tasks (list): all tasks/levels that are useable
        """
        task = "task" + tasks[0][0]

        if num_episodes != None:
            self.rate = num_episodes

        base_dir = os.path.join(self.root_dir, f"{task}/t*")
        for f in glob.glob(base_dir):
            shutil.rmtree(f)

        for key, val in self.distribution.items():
            num_scenarios = math.ceil(self.rate * val)
            print(f"Num of {key} : {num_scenarios}")
            if num_scenarios != 0:
                build_scenarios(
                    task=task,
                    level_name=key,
                    totals={"train": num_scenarios, "test": 1},
                    root_path=self.root_dir,
                    stopwatcher_behavior=None,
                    stopwatcher_route=None,
                    save_dir=self.save_dir,
                )
