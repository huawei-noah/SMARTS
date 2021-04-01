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

import numpy as np
import matplotlib.pyplot as plt

from ultra.scenarios.generate_scenarios import build_scenarios


class CurriculumInfo:
    def __init__(self):
        pass

    @classmethod
    def initialize(cls, curriculum):
        cls.episode_based_toggle = bool(
            curriculum["conditions"]["episode_based"]["toggle"]
        )
        cls.episode_based_cycle = bool(
            curriculum["conditions"]["episode_based"]["cycle"]
        )
        cls.pass_based_toggle = bool(curriculum["conditions"]["pass_based"]["toggle"])
        cls.pass_based_pass_rate = float(
            curriculum["conditions"]["pass_based"]["pass_rate"]
        )
        cls.pass_based_sample_rate = curriculum["conditions"]["pass_based"][
            "sample_rate"
        ]
        # cls.pass_based_sampling_offset = curriculum["conditions"]["pass_based"]["sampling_offset"]

        if cls.episode_based_toggle == cls.pass_based_toggle == True:
            raise Exception(
                "Both condition toggles are set to True. Only one condition should be chosen"
            )
        elif cls.episode_based_toggle == cls.pass_based_toggle == False:
            raise Exception(
                "Both condition toggles are set to False. Please choose one condition"
            )

    # @classmethod
    # def reset_pass_based_sampling_offset(cls, curriculum):
    #     cls.pass_based_sampling_offset = curriculum["conditions"]["pass_based"]["sampling_offset"]

    # @classmethod
    # def decrement_sampling_offset(cls):
    #     cls.pass_based_sampling_offset -= 1


class Coordinator:
    def __init__(self, gb_curriculum_dir, num_episodes):
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
        self.grade_checkpoints = []
        self.num_episodes = num_episodes
        self.grade_size = 0
        self.densities_data = []

        self.overall_densities_counter = {
            "no-traffic": 0,
            "low-density": 0,
            "mid-density": 0,
            "high-density": 0,
        }
        self.grade_densities_counter = {
            "no-traffic": 0,
            "low-density": 0,
            "mid-density": 0,
            "high-density": 0,
        }

        CurriculumInfo.initialize(self.curriculum)

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

    def next_grade(self):
        # Get task and level information
        counter = next(self.counter) + 1
        self.grade = self.curriculum["grades"][counter]

    def get_num_of_grades(self):
        return len(self.curriculum["grades"])

    def get_grade(self):
        return self.grade

    def get_checkpoints(self):
        return f"Episode intervals: {self.grade_checkpoints}"

    def display_grade_scenario_distribution(self):
        print("Traffic density distribution for previous grade:")
        for density in self.grade_densities_counter:
            if self.grade_size != 0:
                print(
                    f"{density}: {round(self.grade_densities_counter[density] / self.grade_size, 2)}"
                )
            else:
                print(f"{density}: 0.0")

    def record_density_data(self, scenario_density):
        self.overall_densities_counter[scenario_density] += 1
        self.grade_densities_counter[scenario_density] += 1
        return self.overall_densities_counter[scenario_density]

    def reset_grade_densities_counter(self):
        self.densities_data.append(self.grade_densities_counter)
        self.grade_densities_counter = {
            "no-traffic": 0,
            "low-density": 0,
            "mid-density": 0,
            "high-density": 0,
        }

    def plot_densities_data(self, file):
        data = []
        print(self.densities_data)
        for grade in self.densities_data:
            temp1 = []
            temp2 = []
            for key, val in grade.items():
                temp2.append(key)
                print("GRADE SIZE:,", self.grade_size)
                temp1.append(round(val / self.grade_size, 2))
            data.append(temp1)

        print("DATA:>>>>>>>>>>>>>>>>>>>:", data)
        X = np.arange(self.get_num_of_grades())
        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.75, 0.75])

        colors = ["g", "m", "b", "r"]
        inc = 0.0
        counter = 0
        if len(data) == 1:
            ax.bar(X + inc, data[counter], color="blue", width=0.25)
        else:
            for color in colors:
                ax.bar(X + inc, data[counter], color=color, width=0.25)
                inc += 0.25
                counter += 1

        ax.set_ylabel("Probability of scenarios in grade")
        ax.set_title("Probability (%)")
        ax.set_xticks(X)

        plt.savefig(file)

    def get_pass_based_sample_rate(self):
        return self.curriculum["conditions"]["pass_based"]["sample_rate"]

    def graduate(self, index, num_episodes, average_scenarios_passed=None):
        """ Conditions on when to graduate """
        if CurriculumInfo.episode_based_toggle:
            return self.episode_based(index, num_episodes)
        elif CurriculumInfo.pass_based_toggle:
            return self.pass_based(index, num_episodes, average_scenarios_passed)

    def episode_based(self, index, num_episodes):
        # Switch to next grade based on number of episodes completed
        self.grade_size = int(num_episodes / self.get_num_of_grades())
        if index == 0:
            self.display()
            self.grade_checkpoints.append(index)
            self.rotation_counter += 1 if CurriculumInfo.episode_based_cycle else 0
        elif (index % int(num_episodes / self.get_num_of_grades())) == 0 and index != 0:
            # Switch grade
            self.next_grade()
            self.display()
            self.reset_grade_densities_counter()
            self.grade_completed = True
            self.grade_checkpoints.append(index)
            self.rotation_counter += 1 if CurriculumInfo.episode_based_cycle else 0
        else:
            self.grade_completed = False

        if self.rotation_counter > self.get_num_of_grades():
            self.cycle_completed = True

        return (self.grade_completed, self.cycle_completed)

    def pass_based(self, index, num_episodes, average_scenarios_passed):
        # Switch to next grade on the basis of certain percentage of completed scenarios
        if index != 0:
            if (
                average_scenarios_passed > CurriculumInfo.pass_based_pass_rate
                and self.rotation_counter <= self.get_num_of_grades()
            ):
                # print(f"({index}) AVERAGE SCENARIOS PASSED: {average_scenarios_passed}")
                # print("pass_based_sampling_offset: ", CurriculumInfo.pass_based_sampling_offset)
                # if CurriculumInfo.pass_based_sampling_offset <= 0:
                self.next_grade()
                self.grade_size += 1
                self.display()
                self.reset_grade_densities_counter()
                self.grade_size = 0
                self.grade_completed = True
                self.rotation_counter += 1
                self.grade_checkpoints.append(index)
            #     CurriculumInfo.reset_pass_based_sampling_offset()
            # CurriculumInfo.decrement_sampling_offset(self.curriculum)
            else:
                self.grade_completed = False
                self.grade_size += 1
        else:
            self.display()
            self.rotation_counter += 1
            self.grade_checkpoints.append(index)

        if self.rotation_counter > self.get_num_of_grades():
            self.cycle_completed = True

        return (self.grade_completed, self.cycle_completed)

    def calculate_average_scenario_passed(
        self, episode, total_scenarios_passed, agents, asp
    ):
        if (episode.index + 1) % CurriculumInfo.pass_based_sample_rate == 0:
            total_scenarios_passed += episode.info[episode.active_tag][
                list(agents.keys())[0]
            ].data["reached_goal"]
            # print(
            #     f"({episode.index + 1}) (SAMPLING) TOTAL SCENARIOS PASSED PER EVAL RATE:",
            #     total_scenarios_passed,
            # )
            average_scenarios_passed = (
                total_scenarios_passed / self.get_pass_based_sample_rate()
            )
            print(
                f"({episode.index + 1}) AVERAGE SCENARIOS PASSED: {average_scenarios_passed}"
            )
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
            self.display_grade_scenario_distribution()
            print("\nGRADE SIZE :", self.grade_size)
            print(f"\nCurrent grade: {self.grade}")
            print("----------------------------------------------------\n")
        except Exception as e:
            print(e)
            pass
