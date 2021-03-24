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
import yaml
from itertools import cycle

from ultra.scenarios.generate_scenarios import build_scenarios


class coordinator:
    def __init__(self, gb_curriculum_dir):
        self.mode = False

        root_dir = gb_curriculum_dir  # Path to grade based config file (config.yaml needed for populating grades with scenarios (tasks, levels))
        base_dir = os.path.join(os.path.dirname(__file__), root_dir)
        grades_dir = os.path.join(base_dir, "config.yaml")

        with open(grades_dir, "r") as task_file:
            self.curriculum = yaml.safe_load(task_file)["curriculum"]

        self.counter = cycle(tuple([i * 1 for i in range(self.get_num_of_grades())]))

        self.cycle_completed = False
        self.grade_completed = False
        self.rotation_counter = 0
        self.plot_arr = []
        self.grade_length = []

    def build_all_scenarios(self, root_path, save_dir):
        for key in self.curriculum["grades"]:
            for task, level in self.curriculum["grades"][key]:
                build_scenarios(
                    task=f"task{task}",
                    level_name=level,
                    root_path=root_path,
                    stopwatcher_behavior=None,
                    stopwatcher_route=None,
                    save_dir=save_dir,
                )

    def next_grade(self, grade):
        # Get task and level information
        self.grade = self.curriculum["grades"][grade]

    def get_num_of_grades(self):
        return len(self.curriculum["grades"])

    def get_grade(self):
        return self.grade

    def get_pass_based_sample_rate(self):
        return self.curriculum["conditions"]["pass_based"]["sample_rate"]

    def graduate(self, index, num_episodes, average_scenarios_passed=None):
        """ Conditions on when to graduate """

        if self.curriculum["conditions"]["episode_based"]["toggle"] == True:
            # Switch to next grade based on number of episodes completed
            if (index % int(num_episodes / self.get_num_of_grades())) == 0:
                self.next_grade(next(self.counter) + 1)
                print(self.display())
                self.rotation_counter += (
                    1 if self.curriculum["conditions"]["episode_based"]["cycle"] else 0
                )
                self.grade_completed = True
            else:
                self.grade_completed = False

            if self.rotation_counter > self.get_num_of_grades():
                self.cycle_completed = True

            return (self.grade_completed, self.cycle_completed)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        # Switch to next grade on the basis of certain percentage of completed scenarios

        if self.curriculum["conditions"]["pass_based"]["toggle"] == True:
            if index != 0:
                if (
                    average_scenarios_passed
                    > float(self.curriculum["conditions"]["pass_based"]["pass_rate"])
                    and self.rotation_counter <= self.get_num_of_grades()
                ):
                    # print(f"({index}) AVERAGE SCENARIOS PASSED: {average_scenarios_passed}")
                    self.next_grade(next(self.counter) + 1)
                    print(self.display())
                    self.grade_completed = True
                    self.rotation_counter += 1
                    self.grade_length.append(index)
                else:
                    self.grade_completed = False
            else:
                self.next_grade(next(self.counter) + 1)
                print(self.display())
                self.grade_completed = True
                self.rotation_counter += 1
                self.grade_length.append(index)

            if self.rotation_counter > self.get_num_of_grades():
                self.cycle_completed = True

            if self.cycle_completed:
                print("Average scenario success array: ", self.plot_arr)
                print("Epsiode intervals: ", self.grade_length)

            return (self.grade_completed, self.cycle_completed)

    def display(self):
        return f"\nCurrent grade: {self.grade}\n"
