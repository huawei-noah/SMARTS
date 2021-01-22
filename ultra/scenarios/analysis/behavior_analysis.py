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
import numpy as np
import gym, random, re, timeit, sys, ray, torch
import glob, os, argparse, json
from matplotlib import pyplot as plt
import dill
from collections import defaultdict
from smarts.core.utils.episodes import episodes
from ultra.scenarios.common.visualization import (
    draw_intersection,
    convert_to_gif,
    profile_vehicles,
)
from smarts.core.agent import AgentSpec, Agent
from smarts.core.controllers import ActionSpaceType
from smarts.core.agent_interface import (
    AgentInterface,
    AgentType,
    NeighborhoodVehicles,
)
from ultra.scenarios.common.social_vehicle_definitions import get_social_vehicle_color
from ultra.scenarios.generate_scenarios import get_direction
from ultra.scenarios.analysis.base_analysis import BaseAnalysis

num_gpus = 1 if torch.cuda.is_available() else 0


@ray.remote(num_gpus=num_gpus / 2)
class BehaviorAnalysis(BaseAnalysis):
    def __init__(self):
        super(BaseAnalysis, self).__init__()
        self.analysis = []

    def draw_plots(self, save_dir, failure_time=None):
        start = 0
        end = 600
        behaviors = {}
        for scenario in self.analysis:
            for behavior in scenario.keys():
                if behavior not in behaviors:
                    behaviors[behavior] = {
                        "avg_speed_junction": [],
                        "min_speed_junction": [],
                        "max_speed_junction": [],
                        "avg_accel_junction": [],
                        "min_accel_junction": [],
                        "max_accel_junction": [],
                        "total_steps_junction": [],
                        "exceeds_speed_limit": 0,
                    }
                s, e = scenario[behavior]["in_junction"][0]
                behaviors[behavior]["avg_speed_junction"].append(
                    np.mean(scenario[behavior]["speeds"][s:e])
                )
                behaviors[behavior]["min_speed_junction"].append(
                    scenario[behavior]["min_speed_junction"][0]
                )
                behaviors[behavior]["max_speed_junction"].append(
                    scenario[behavior]["max_speed_junction"][0]
                )
                behaviors[behavior]["avg_accel_junction"].append(
                    np.mean(scenario[behavior]["accels"][s:e])
                )
                behaviors[behavior]["min_accel_junction"].append(
                    scenario[behavior]["min_accel_junction"][0]
                )
                behaviors[behavior]["max_accel_junction"].append(
                    scenario[behavior]["max_accel_junction"][0]
                )
                behaviors[behavior]["total_steps_junction"].append(
                    scenario[behavior]["steps_in_junction"][0]
                )

        for key in [
            "min_speed_junction",
            "max_speed_junction",
            "avg_speed_junction",
            "min_accel_junction",
            "max_accel_junction",
            "avg_accel_junction",
            "total_steps_junction",
        ]:
            plt.figure()
            if "speed" in key:
                bins = [i for i in range(40)]
            elif "accel" in key:
                bins = [i for i in range(-15, 15)]
            else:
                bins = [i for i in range(1000)]
            for behavior, data in behaviors.items():
                n, bins, patches = plt.hist(
                    x=data[key], bins=bins, color="#0504aa", alpha=0.7, rwidth=0.85
                )
                plt.grid(axis="y", alpha=0.75)
                plt.xlabel(key)
                plt.title(behavior)
                plt.ylabel("Frequency")
                maxfreq = n.max()
                # Set a clean upper y-axis limit.
                plt.ylim(
                    ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10
                )
                plt.savefig(f"{save_dir}/{behavior}_{key}.png")
                plt.close()

    def analyze(self, episode_time, stopwatcher_max_steps, simulation_data):
        analysis = {}
        for v_id, v_state in self.social_vehicles_states.items():
            if "stopwatcher" in v_state["behavior"]:
                _, behavior = v_state["behavior"].split("_")
                if behavior not in self.analysis:
                    analysis[behavior] = {
                        "max_steps": [],
                        "steps_in_junction": [],
                        "max_road_speed": None,
                        "speeds": [],
                        "accels": [],
                        "min_speed_junction": [],
                        "max_speed_junction": [],
                        "min_accel_junction": [],
                        "max_accel_junction": [],
                        "in_junction": [],
                        "steps_in_junction": [],
                    }
                in_junction = v_state["in_junction"]
                speed_in_junction = v_state["speeds"][in_junction[0] : in_junction[1]]
                accel_in_junction = v_state["accels"][in_junction[0] : in_junction[1]]
                analysis[behavior]["in_junction"].append(in_junction)
                analysis[behavior]["min_speed_junction"].append(min(speed_in_junction))
                analysis[behavior]["max_speed_junction"].append(max(speed_in_junction))
                analysis[behavior]["min_accel_junction"].append(min(accel_in_junction))
                analysis[behavior]["max_accel_junction"].append(max(accel_in_junction))
                print("in junction", in_junction)
                analysis[behavior]["max_steps"].append(v_state["steps"])
                analysis[behavior]["steps_in_junction"].append(
                    int(in_junction[1]) - int(in_junction[0])
                )
                analysis[behavior][
                    "max_road_speed"
                ] = 19.44  # road_speads[v_state["route"]]
                analysis[behavior]["speeds"] = v_state["speeds"]
                analysis[behavior]["accels"] = v_state["accels"]
        self.analysis.append(analysis)

    def save_data(self, save_dir):
        with open(f"{save_dir}/analysis.pkl", "wb") as handle:
            dill.dump(self.analysis, handle)
        print(f"saved data to {save_dir}/analysis.pkl")

    def load_data(self, path):
        with open(path, "rb") as handle:
            self.analysis = dill.load(handle)
        print(f"loaded data {len(self.analysis)} from {path}")

    def run(self, **params):
        super().run(**params, analyze_func=self.analyze)
