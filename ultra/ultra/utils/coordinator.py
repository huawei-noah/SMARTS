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
import yaml, csv, glob, shutil
from itertools import cycle
import copy

import numpy as np
import matplotlib.pyplot as plt

from ultra.scenarios.generate_scenarios import build_scenarios


class CurriculumInfo:
    def __init__(self):
        pass

    @classmethod
    def initialize(cls, gb_curriculum_dir):
        root_dir = gb_curriculum_dir  # Path to grade based config file (config.yaml needed for populating grades with scenarios (tasks, levels))
        base_dir = os.path.join(os.path.dirname(__file__), root_dir)
        grades_dir = os.path.join(base_dir, "config.yaml")

        with open(grades_dir, "r") as task_file:
            cls.curriculum = yaml.safe_load(task_file)["curriculum"]

        cls.episode_based_toggle = bool(
            cls.curriculum["conditions"]["episode_based"]["toggle"]
        )
        cls.episode_based_cycle = bool(
            cls.curriculum["conditions"]["episode_based"]["cycle"]
        )
        cls.pass_based_toggle = bool(
            cls.curriculum["conditions"]["pass_based"]["toggle"]
        )
        cls.pass_based_pass_rate = float(
            cls.curriculum["conditions"]["pass_based"]["pass_rate"]
        )
        cls.pass_based_sample_rate = cls.curriculum["conditions"]["pass_based"][
            "sample_rate"
        ]
        cls.pass_based_warmup_episodes = int(
            cls.curriculum["conditions"]["pass_based"]["warmup_episodes"]
        )
        cls.eval_per_grade = bool(cls.curriculum["conditions"]["eval_per_grade"])

        if cls.episode_based_toggle == cls.pass_based_toggle == True:
            raise Exception(
                "Both condition toggles are set to True. Only one condition should be chosen"
            )
        elif cls.episode_based_toggle == cls.pass_based_toggle == False:
            raise Exception(
                "Both condition toggles are set to False. Please choose one condition"
            )


class ScenarioDataHandler:
    def __init__(self, tag):
        self.overall_densities_counter = {
            "no-traffic": 0,
            "low-density": 0,
            "mid-density": 0,
            "high-density": 0,
        }
        self.grade_densities_counter = copy.deepcopy(self.overall_densities_counter)
        self.densities_data = []
        self.tag = tag

    def record_density_data(self, scenario_density):
        if scenario_density != "p-test":
            self.overall_densities_counter[scenario_density] += 1
            self.grade_densities_counter[scenario_density] += 1
            return self.overall_densities_counter[scenario_density]
        return

    def save_grade_density(self, grade_size):
        temp = []
        print(f"({self.tag}) Grade size: {grade_size}")
        for density in self.grade_densities_counter:
            if grade_size != 0:
                temp.append(
                    round(self.grade_densities_counter[density] / grade_size, 2)
                )
            else:
                pass
        self.densities_data.append(temp)
        self.grade_densities_counter = {
            "no-traffic": 0,
            "low-density": 0,
            "mid-density": 0,
            "high-density": 0,
        }

    def display_grade_scenario_distribution(self, grade_size, grade=None):
        if grade == None:
            grade = ">>> No grades <<<"
        print("----------------------------------------------------")
        print(f"Traffic density distribution for {grade} (or {self.tag} run):")
        for density in self.grade_densities_counter:
            if grade_size != 0:
                print(
                    f"{density}: {round(self.grade_densities_counter[density] / grade_size, 2)}"
                )
            else:
                print(f"{density}: 0.0")
        print("----------------------------------------------------\n")

    def plot_densities_data(self, filepath=None):
        total_density_data = self.densities_data
        # print(total_density_data)
        header = ["no-traffic", "low", "mid", "high"]
        header.insert(0, "")

        with open(filepath, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(header)
            counter = 0
            for i in range(len(total_density_data)):
                total_density_data[i].insert(0, f"grade-{i}")
                writer.writerow(total_density_data[i])
        header = []


class Coordinator:
    def __init__(self, gb_curriculum_dir, num_episodes):
        CurriculumInfo.initialize(gb_curriculum_dir)

        self.mode = False
        self.counter = cycle(tuple([i * 1 for i in range(self.get_num_of_grades())]))
        self.eval_counter = cycle(tuple([i * 1 for i in range(self.get_num_of_grades())]))
        self.grade_checkpoints = []
        self.num_episodes = num_episodes
        self.grade_counter = 0
        self.episode_per_grade = 1
        self.warmup_episodes = 1
        self.end_warmup = False
        self.eval_check = False
        self.eval_per_grade = CurriculumInfo.eval_per_grade

    def build_all_scenarios(self, root_path, save_dir):
        for key in CurriculumInfo.curriculum["grades"]:
            for task, level in CurriculumInfo.curriculum["grades"][key]:
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
        self.grade = CurriculumInfo.curriculum["grades"][counter]
    
    def next_eval_grade(self):
        # Get task and level information
        counter = next(self.eval_counter) + 1
        self.eval_grade = CurriculumInfo.curriculum["grades"][counter]

    def get_num_of_grades(self):
        return len(CurriculumInfo.curriculum["grades"])

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
            # print(
            #     f"({episode.index + 1}) (SAMPLING) TOTAL SCENARIOS PASSED PER EVAL RATE:",
            #     total_scenarios_passed,
            # )
            average_scenarios_passed = total_scenarios_passed / sample_rate
            # print(
            #     f"({episode.index + 1}) AVERAGE SCENARIOS PASSED: {average_scenarios_passed}"
            # )
            total_scenarios_passed = 0.0
            return average_scenarios_passed, total_scenarios_passed
        else:
            total_scenarios_passed += episode.info[episode.active_tag][
                list(agents.keys())[0]
            ].data["reached_goal"]
            # print(
            #     f"({episode.index + 1}) TOTAL SCENARIOS PASSED PER EVAL RATE:",
            #     total_scenarios_passed,
            # )
            return asp, total_scenarios_passed

    def display(self):
        try:
            print("\n----------------------------------------------------")
            print("Grade counter :", self.grade_counter)
            print(f"\nCurrent grade: {self.grade}")
            print("----------------------------------------------------")
        except Exception as e:
            print(e)
            pass

class DynamicScenarios():
    def __init__(self, root_dir, save_dir, rate=None):
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
        if level == "grade1":
            control_density = "no-traffic"
        elif level == "grade2":
            control_density = "low-density"
        elif level == "grade3":
            control_density = "mid-density"
        elif level == "grade4":
            control_density = "high-density"

        print("Old distrbution:", self.distribution)

        if increment_mode:
            for key, value in self.distribution.items():
                if control_density == key:
                    self.distribution[key] -= 0.02
                else:
                    self.distribution[key] += 0.02
        
        print("New distrbution:", self.distribution)
    
    def reset_scenario_pool(self):
        base_dir = os.path.join(self.root_dir, "taskgb/t*")
        print(base_dir)
        for f in glob.glob(base_dir):
            shutil.rmtree(f)

        for key, val in self.distribution.items():
            num_scenarios = (self.rate * val)
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

    def sampler(episode, total_scenarios_passed, average_scenarios_passed):
        if self.rate is not None:
            sample_rate = self.rate
        else:
            sample_rate = CurriculumInfo.pass_based_sample_rate

        asp_list = []

        (
            average_scenarios_passed,
            total_scenarios_passed,
        ) = Coordinator.calculate_average_scenario_passed(
            episode, total_scenarios_passed, agents, average_scenarios_passed
        )

        if (
            episode.index + 1
        ) % sample_rate == 0:  # Set sample rate (flag needs to be set)
            print(
                f"({episode.index + 1}) AVERAGE SCENARIOS PASSED: {average_scenarios_passed}"
            )
            asp_list.append(
                tuple((episode.index + 1, average_scenarios_passed))
            )
            self.change_distribution()
            self.reset_scenario_pool()

        return asp_list
