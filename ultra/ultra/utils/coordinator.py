import os, sys
import yaml
from ultra.scenarios.generate_scenarios import build_scenarios


class coordinator:
    def __init__(self, root_dir):
        # self._task = task
        # self._level = ""
        base_dir = os.path.join(os.path.dirname(__file__), root_dir)
        grades_dir = os.path.join(base_dir, "config.yaml")

        with open(grades_dir, "r") as task_file:
            self.curriculum = yaml.safe_load(task_file)["grades"]

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

    def get_num_of_grades(self):
        return len(self.curriculum)

    def get_grade(self):
        return self.grade

    def __str__(self):
        return f"\nCurrent grade: {self.grade}\n"
