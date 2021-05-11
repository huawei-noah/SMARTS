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
        """Initializes curriculum and sets up variables

        Args:
            curriculum_path (str): path to curriculum config file
            num_episodes (int): total number of episodes
        """
        CurriculumInfo.initialize(curriculum_dir)

        self._train_counter = cycle(tuple([i * 1 for i in range(self.num_grades)]))
        self._eval_counter = cycle(tuple([i * 1 for i in range(self.num_grades)]))
        self._num_episodes = num_episodes
        self._grade_counter = 0
        self._episodes_per_grade = -1
        self._warmup_episodes = 1
        self.end_warmup = False

    def build_all_scenarios(self, root_path, save_dir):
        """Builds all of the scenarios needed in the curriculum

        Args:
            root_path (str): path to tasks' directory
            save_dir (str): path to save the scenarios
        """
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

    def next_train_grade(self):
        """Increments training grade counter and stores new training grade inside
        global grade carrier
        """
        counter = next(self._train_counter) + 1
        self._train_grade = CurriculumInfo.curriculum["static"]["grades"][counter]

    @property
    def train_grade(self) -> list:
        """
        Returns:
            current train grade
        """
        return self._train_grade

    def next_eval_grade(self):
        """Increments evaluation grade counter and stores evaluation grade inside
        global grade carrier. Coordinator needs two methods to control training
        grade and evaluation grade. This is due to the train/eval parallelization,
        which causes the coordinator to work in training/evaluation at different
        times. To avoid the mix up of grades, there are separate methods to deal
        with the train/eval.
        """
        # Get task and level information
        counter = next(self._eval_counter) + 1
        self._eval_grade = CurriculumInfo.curriculum["static"]["grades"][counter]

    @property
    def eval_grade(self) -> list:
        """
        Returns:
            current eval grade
        """
        return self._eval_grade

    @property
    def num_grades(self):
        """
        Returns:
            the number of grades in the static curriculum
        """
        return len(CurriculumInfo.curriculum["static"]["grades"])

    @property
    def grade_size(self) -> int:
        """
        Returns:
            the number of episodes in the current grade
        """
        if CurriculumInfo.episode_based_toggle:
            return int(self._num_episodes / self.num_grades)
        else:
            return self._episodes_per_grade

    def check_cycle_condition(self, index) -> bool:
        """Checks if curriculum needs to be repeated

        Args:
            index (int): reference to an episode in an sequence of episodes

        Returns:
            exit signal to stop grades from starting over agent
        """
        if (CurriculumInfo.episode_based_cycle == False) and (
            CurriculumInfo.episode_based_toggle == True
        ):
            if index + 1 > int(self.num_grades * self.grade_size):
                return True
        elif (CurriculumInfo.pass_based_toggle == True) and (
            self._grade_counter >= self.num_grades
        ):
            return True
        return False

    def graduate(self, index, average_reached_goal=None) -> bool:
        """A registar for agents to check if they are eligible
            to enter into the next grade

        Args:
            index (int): reference to an episode in an sequence of episodes
            average_reached_goal (float): mean value of episodes that the
                                          agent has completed over some
                                          sampling rate

        Returns:
            the eligibility of the agents to enter the next grade
        """
        self._episodes_per_grade += 1
        if CurriculumInfo.pass_based_toggle == True:
            if CurriculumInfo.pass_based_warmup_episodes != 0:
                if (
                    self._warmup_episodes % CurriculumInfo.pass_based_warmup_episodes
                    == 0
                ) and (self.end_warmup == False):
                    print("***WARM-UP episode:", self._warmup_episodes)
                    self._warmup_episodes = 1
                    self.end_warmup = True
                    return False
                elif self.end_warmup == False:
                    print("***WARM-UP episode:", self._warmup_episodes)
                    self._warmup_episodes += 1
                    return False
            else:
                self.end_warmup = True

            if (
                self.end_warmup == True
                or CurriculumInfo.pass_based_warmup_episodes == 0
            ):
                if CurriculumInfo.pass_based_toggle:
                    return self.pass_based(index, average_reached_goal)

        if CurriculumInfo.episode_based_toggle:
            return self.episode_based(index)

    def episode_based(self, index) -> bool:
        """Graduates the agent after completing fix amount of
            episodes n. n is the total episodes divided the
            total number of grades

        Args:
            index (int): reference to an episode in an sequence of episodes

        Returns:
            the eligibility of the agents to enter the next grade
        """
        if index == 0:
            self._grade_counter += 1
            self.display()
        elif (index % int(self._num_episodes / self.num_grades)) == 0 and index != 0:
            self.next_train_grade()
            self.display()
            self._grade_counter += 1
            return True
        else:
            return False

    def pass_based(self, index, average_reached_goal) -> bool:
        """Graduates the agent after it has obtained an
            average reached goal greater than or equal to
            the specified pass rate

        Args:
            index (int): reference to an episode in an sequence of episodes
            average_reached_goal (float): mean value of episodes that the
                                          agent has completed over some
                                          sampling rate

        Returns:
            the eligibility of the agents to enter the next grade
        """
        # Switch to next grade on the basis of certain percentage of completed scenarios
        if index != 0:
            if average_reached_goal >= CurriculumInfo.pass_based_pass_rate:
                print(f"({index}) AVERAGE REACHED GOAL: {average_reached_goal}")
                self.next_train_grade()
                self.display()
                self._grade_counter += 1
                return True
            else:
                return False
        else:
            self.display()

    def display(self):
        """Prints information about succeeding grade"""
        print("\n----------------------------------------------------")
        print("Grade counter :", self._grade_counter)
        print(f"\nCurrent grade: {self._train_grade}")
        print("----------------------------------------------------")
