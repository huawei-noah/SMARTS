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
    def __init__(self, root_dir):
        self.mode = False

        base_dir = os.path.join(os.path.dirname(__file__), root_dir)
        grades_dir = os.path.join(base_dir, "config.yaml")

        with open(grades_dir, "r") as task_file:
            self.curriculum = yaml.safe_load(task_file)["grades"]

        self.counter = cycle(tuple([i * 1 for i in range(self.get_num_of_grades())]))

    def set_grade_mode(self, mode):
        self.mode = mode

    def get_grade_mode(self):
        return self.mode

    def build_all_scenarios(self):
        for key in self.curriculum:
            for task, level in self.curriculum[key]:
                build_scenarios(
                    task=f"task{task}",
                    level_name=level,
                    root_path="ultra/scenarios",
                    stopwatcher_behavior=None,
                    stopwatcher_route=None,
                    save_dir=None,
                )

    def next_grade(self, grade):
        # Get task and level information
        self.grade = self.curriculum[grade]

    def graduate(self, index, num_episodes, agent_id, average_scenarios_passed):
        """ Conditions on when to graduate """

        # # Switch to next grade based on number of episodes completed
        # if (episode.index % (num_episodes / self.get_num_of_grades())) == 0:
        #     self.next_grade(next(self.counter) + 1)
        #     return True
        # else:
        #     return False

        # Switch to next grade on the basis of certain percentage of completed scenarios
        if index != 0:
            print("\nAVERAGE SCENARIOS PASSED:", average_scenarios_passed)
            if average_scenarios_passed > 0.50:
                self.next_grade(next(self.counter) + 1)
                return True
            else:
                return False
        else:
            # Initial switch (admission into curriculum)
            self.next_grade(next(self.counter) + 1)
            return True

    def get_num_of_grades(self):
        return len(self.curriculum)

    def get_grade(self):
        return self.grade

    def __str__(self):
        return f"\nCurrent grade: {self.grade}\n"
