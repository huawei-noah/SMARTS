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
import csv
import copy

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

    def plot_densities_data(self, filepath=None, grade_mode=False):
        total_density_data = self.densities_data
        # print(total_density_data)
        header = ["no-traffic", "low", "mid", "high"]
        header.insert(0, "")

        with open(filepath, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(header)
            counter = 0
            for i in range(len(total_density_data)):
                if not grade_mode:
                    total_density_data[i].insert(0, f"sample-{i}")
                else:
                    total_density_data[i].insert(0, f"grade-{i}")
                writer.writerow(total_density_data[i])
        header = []