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
from itertools import cycle

from ultra.scenarios.generate_scenarios import build_scenarios
from ultra.utils.curriculum.curriculum_info import CurriculumInfo


class Coordinator:
    def __init__(self, curriculum_dir, num_episodes):
        CurriculumInfo.initialize(curriculum_dir)

        self.mode = False
        self.counter = cycle(tuple([i * 1 for i in range(self.get_num_of_grades())]))
        self.eval_counter = cycle(
            tuple([i * 1 for i in range(self.get_num_of_grades())])
        )
        self.grade_checkpoints = []
        self.num_episodes = num_episodes
        self.grade_counter = 0
        self.episode_per_grade = 1
        self.warmup_episodes = 1
        self.end_warmup = False
        self.eval_check = False
        self.eval_per_grade = CurriculumInfo.eval_per_grade

    def build_all_scenarios(self, root_path, save_dir):
        for key in CurriculumInfo.curriculum["static"]["grades"]:
            for task, level in CurriculumInfo.curriculum["static"]["grades"][key]:
                build_scenarios(
                    task=f"task{task}",
                    level_name=level,
                    root_path=root_path,
                    stopwatcher_behavior=None,
                    stopwatcher_route=None,
                    save_dir=save_dir,
                )

    def next_grade(self):
        # Get task and level information
        counter = next(self.counter) + 1
        self.grade = CurriculumInfo.curriculum["static"]["grades"][counter]

    def next_eval_grade(self):
        # Get task and level information
        counter = next(self.eval_counter) + 1
        self.eval_grade = CurriculumInfo.curriculum["static"]["grades"][counter]

    def get_num_of_grades(self):
        return len(CurriculumInfo.curriculum["static"]["grades"])

    def get_grade(self):
        return self.grade

    def get_eval_grade(self):
        return self.eval_grade

    def get_grade_size(self):
        if CurriculumInfo.episode_based_toggle:
            return int(self.num_episodes / self.get_num_of_grades())
        else:
            grade_size = self.episode_per_grade
            return grade_size

    def check_cycle_condition(self, index):
        if (CurriculumInfo.episode_based_cycle == False) and (
            CurriculumInfo.episode_based_toggle == True
        ):
            if index + 1 > int(self.get_num_of_grades() * self.get_grade_size()):
                return True
        elif (CurriculumInfo.pass_based_toggle == True) and (
            self.grade_counter >= self.get_num_of_grades()
        ):
            return True
        return False

    def get_checkpoints(self):
        return f"Episode intervals: {self.grade_checkpoints}"

    def set_eval_check_condition(self, condition: bool):
        self.eval_check = condition

    def get_eval_check_condition(self):
        return self.eval_check

    def graduate(self, index, average_scenarios_passed=None):
        """ Conditions on when to graduate """
        # print("GRADE size counter:", self.episode_per_grade)
        self.episode_per_grade += 1
        if CurriculumInfo.pass_based_toggle == True:
            if CurriculumInfo.pass_based_warmup_episodes != 0:
                if (
                    self.warmup_episodes % CurriculumInfo.pass_based_warmup_episodes
                    == 0
                ) and (self.end_warmup == False):
                    print("***WARM-UP episode:", self.warmup_episodes)
                    self.warmup_episodes = 1
                    self.end_warmup = True
                    return False
                elif self.end_warmup == False:
                    print("***WARM-UP episode:", self.warmup_episodes)
                    self.warmup_episodes += 1
                    return False
            else:
                self.end_warmup = True

            if (
                self.end_warmup == True
                or CurriculumInfo.pass_based_warmup_episodes == 0
            ):
                if CurriculumInfo.pass_based_toggle:
                    return self.pass_based(index, average_scenarios_passed)

        if CurriculumInfo.episode_based_toggle:
            return self.episode_based(index)

    def episode_based(self, index):
        # Switch to next grade based on number of episodes completed
        if index == 0:
            self.grade_counter += 1
            self.display()
            self.grade_checkpoints.append(index)
        elif (
            index % int(self.num_episodes / self.get_num_of_grades())
        ) == 0 and index != 0:
            # Switch grade
            self.next_grade()
            self.grade_counter += 1
            self.display()
            self.grade_checkpoints.append(index)
            return True
        else:
            return False

    def pass_based(self, index, average_scenarios_passed):
        # Switch to next grade on the basis of certain percentage of completed scenarios
        if index != 0:
            if average_scenarios_passed >= CurriculumInfo.pass_based_pass_rate:
                print(f"({index}) AVERAGE SCENARIOS PASSED: {average_scenarios_passed}")
                self.next_grade()
                self.grade_counter += 1
                # self.episode_per_grade = 0
                self.display()
                self.grade_checkpoints.append(index)
                return True
            else:
                return False
        else:
            self.display()
            self.grade_checkpoints.append(index)

    @staticmethod
    def calculate_average_scenario_passed(
        episode, total_scenarios_passed, agents, asp, rate=None
    ):
        try:
            sample_rate = CurriculumInfo.pass_based_sample_rate
        except AttributeError as e:
            sample_rate = rate

        if (episode.index + 1) % sample_rate == 0:
            total_scenarios_passed += episode.info[episode.active_tag][
                list(agents.keys())[0]
            ].data["reached_goal"]
            average_scenarios_passed = total_scenarios_passed / sample_rate
            total_scenarios_passed = 0.0
            return average_scenarios_passed, total_scenarios_passed
        else:
            total_scenarios_passed += episode.info[episode.active_tag][
                list(agents.keys())[0]
            ].data["reached_goal"]

    def display(self):
        try:
            print("\n----------------------------------------------------")
            print("Grade counter :", self.grade_counter)
            print(f"\nCurrent grade: {self.grade}")
            print("----------------------------------------------------")
        except Exception as e:
            print(e)
            pass
