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


class CurriculumInfo:
    def __init__(self):
        """ Empty constructor """
        pass

    @classmethod
    def initialize(cls, curriculum_path=None):
        """Retrieves curriculum information from curriculum config
        file.

        Args:
            curriculum_path (str): path to curriculum config file

        Raises:
            ValueError: If static_toggle and dynamic_toggle are both true
                        or false, and if static_conditions are both true
                        or false
        """
        if curriculum_path is None:
            curriculum_path = "../../scenarios/curriculum/"  # Default path to curriculum path in ultra/scenarios
        else:
            curriculum_path = os.path.join("../../", curriculum_path)

        root_dir = curriculum_path  # Path to curriculum config file
        base_dir = os.path.join(os.path.dirname(__file__), root_dir)
        grades_dir = os.path.join(base_dir, "config.yaml")

        with open(grades_dir, "r") as task_file:
            cls.curriculum = yaml.safe_load(task_file)["curriculum"]

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Static curriculum ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        cls.static_curriculum_toggle = bool(cls.curriculum["static"]["toggle"])
        cls.episode_based_toggle = bool(
            cls.curriculum["static"]["conditions"]["episode_based"]["toggle"]
        )
        cls.episode_based_cycle = bool(
            cls.curriculum["static"]["conditions"]["episode_based"]["cycle"]
        )
        cls.pass_based_toggle = bool(
            cls.curriculum["static"]["conditions"]["pass_based"]["toggle"]
        )
        cls.pass_based_pass_rate = float(
            cls.curriculum["static"]["conditions"]["pass_based"]["pass_rate"]
        )
        cls.pass_based_sample_rate = cls.curriculum["static"]["conditions"][
            "pass_based"
        ]["sample_rate"]
        cls.pass_based_warmup_episodes = int(
            cls.curriculum["static"]["conditions"]["pass_based"]["warmup_episodes"]
        )
        cls.eval_per_grade = bool(
            cls.curriculum["static"]["conditions"]["eval_per_grade"]
        )

        if cls.episode_based_toggle == cls.pass_based_toggle == True:
            raise Exception(
                "Both condition toggles are set to True. Only one condition should be chosen"
            )
        elif cls.episode_based_toggle == cls.pass_based_toggle == False:
            raise Exception(
                "Both condition toggles are set to False. Please choose one condition"
            )

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Dynamic curriculum ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        cls.dynamic_curriculum_toggle = bool(cls.curriculum["dynamic"]["toggle"])
        cls.tasks_levels_used = list(cls.curriculum["dynamic"]["tasks_levels_used"])
        cls.sampling_rate = cls.curriculum["dynamic"]["sampling_rate"]

        if cls.static_curriculum_toggle == cls.dynamic_curriculum_toggle == True:
            raise ValueError(
                "Both condition toggles are set to True. Only one condition should be chosen"
            )
        elif cls.static_curriculum_toggle == cls.dynamic_curriculum_toggle == False:
            raise ValueError(
                "Both condition toggles are set to False. Please choose one condition"
            )
