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
import gym, random, re, timeit, copy
import glob, os, argparse, json, ray, torch
from matplotlib import pyplot as plt
import dill
from collections import defaultdict
from smarts.core.utils.episodes import episodes
from smarts.core.agent import AgentSpec, Agent
from smarts.core.controllers import ActionSpaceType
from smarts.core.agent_interface import (
    AgentInterface,
    AgentType,
    NeighborhoodVehicles,
)
from ultra.scenarios.common.visualization import (
    draw_intersection,
    convert_to_gif,
    profile_vehicles,
)
from ultra.scenarios.common.social_vehicle_definitions import get_social_vehicle_color
from ultra.scenarios.analysis.base_analysis import BaseAnalysis

num_gpus = 1 if torch.cuda.is_available() else 0


@ray.remote(num_gpus=num_gpus / 2)
class ScenarioAnalysis(BaseAnalysis):
    def __init__(self):
        super(BaseAnalysis, self).__init__()
        self.analysis = {
            "stopwatcher_max_steps": [],
            "max_num_step_overall_routes": [],
            "steps_taken_by_behavior": defaultdict(lambda: []),
            "steps_taken_by_route": defaultdict(lambda: []),
            "stops_by_route": defaultdict(lambda: []),
            "episode_number_of_steps": [],
            "vehicle_states": [],
            "stop_percentage": defaultdict(lambda: []),
        }
        self.all_simulation_data = []

    def log_steps_summary(self, file_name):
        out_str = []
        all_vehicle_states = []
        for state in self.analysis["vehicle_states"]:
            all_vehicle_states.extend(list(state.values()))
        all_steps = []
        for state in all_vehicle_states:
            if state["route"] == ("south-SN", "west-EW"):
                print(state["end_step"] - state["start_step"])
                all_steps.append(state["end_step"] - state["start_step"])
        all_steps = np.asarray(all_steps)
        steps_thresholds = [100, 200, 500, 1000]
        percentage = [
            np.sum(all_steps > threshold) / len(all_steps)
            for threshold in steps_thresholds
        ]
        out_str.append(f"Step south-west threshold: \t{steps_thresholds}")
        out_str.append(f"Threshold south-west percentage: \t{percentage}")
        with open(file_name, "w") as f:
            f.writelines("\n".join(out_str))

    def analyze(self, episode_step, stopwatcher_max_steps, simulation_data):
        max_step = 0
        for v_id, v_state in self.social_vehicles_states.items():
            step_taken = v_state["end_step"] - v_state["start_step"]
            self.analysis["steps_taken_by_route"][v_state["route"]].append(step_taken)
            self.analysis["steps_taken_by_behavior"][v_state["behavior"]].append(
                step_taken
            )
            self.analysis["stops_by_route"][v_state["route"]].append(
                v_state["stop_step"]  # / time_taken
            )
            if v_state["route"] != ("south-SN", "west-EW"):
                max_time = max(max_step, step_taken)
        self.analysis["episode_number_of_steps"].append(episode_step)
        self.analysis["max_num_step_overall_routes"].append(max_step)
        self.analysis["stopwatcher_max_steps"].append(stopwatcher_max_steps)
        for v_r in self.analysis["stops_by_route"].keys():
            without_zeros = [i for i in self.analysis["stops_by_route"][v_r] if i != 0]
            percent = len(without_zeros) / len(self.analysis["stops_by_route"][v_r])
            self.analysis["stop_percentage"][v_r].append(percent)

        self.all_simulation_data.append(simulation_data)
        self.analysis["vehicle_states"].append(
            copy.deepcopy(self.social_vehicles_states)
        )

        # self.analysis['all_stop_percentage'] = defaultdict(lambda: [])
        # for episode_vs in self.analysis["vehicle_states"]:
        #     episode_stop_steps = defaultdict(lambda: [])
        #     for vs in episode_vs.values():
        #         episode_stop_steps[vs["route"]].append(vs["stop_steps"])
        #     for k, v in episode_stop_steps.items():
        #         without_zeros = [e for e in v if e > 0]
        #         self.analysis['all_stop_percentage'][k].append(len(without_zeros) / len(v))

    def draw_plots(self, save_dir, failure_time):
        self.generate_figures1(save_dir=f"{save_dir}/analysis1")
        self.generate_figures2(
            save_dir=f"{save_dir}/analysis2", failure_time=failure_time
        )
        self.log_steps_summary(f"{save_dir}/step_summary.txt")
        print(f"Plots/results saved in {save_dir}")

    def generate_figures1(self, save_dir):
        print("Generating Analysis 1")
        os.makedirs(save_dir, exist_ok=True)
        #         self.save_histogram(
        #             self.analysis["max_num_step_overall_routes"],
        #             f"{save_dir}/max_num_step_overall_routes.png",
        #             "max_num_step_overall_routes",
        #         )
        self.save_histogram(
            self.analysis["stopwatcher_max_steps"],
            f"{save_dir}/stopwatcher_max_steps.png",
            "stopwatcher_max_steps",
            (0, 6000),
        )

        self.save_histogram(
            self.analysis["episode_number_of_steps"],
            f"{save_dir}/step_episode.png",
            "time step_episode",
            (0, 6000),
        )

        for v_t in self.analysis["steps_taken_by_behavior"].keys():
            self.save_histogram(
                self.analysis["steps_taken_by_behavior"][v_t],
                f"{save_dir}/steps_taken_by_{v_t}.png",
                f"steps {v_t}",
                (0, 6000),
            )
        for v_r in self.analysis["steps_taken_by_route"].keys():
            self.save_histogram(
                self.analysis["steps_taken_by_route"][v_r],
                f"{save_dir}/steps_taken_by_route{v_r}.png",
                f"steps {v_r}",
                (0, 6000),
            )
        self.save_histogram(
            sum(list(self.analysis["steps_taken_by_behavior"].values()), []),
            f"{save_dir}/steps_taken_by_behavior.png",
            "steps all",
            (0, 6000),
        )
        for v_r in self.analysis["stops_by_route"].keys():
            without_zeros = [i for i in self.analysis["stops_by_route"][v_r] if i != 0]
            self.save_histogram(
                without_zeros,
                f"{save_dir}/stops_by_route_{v_r}.png",
                f"stop % {v_r}",
                (0, 6000),
            )
        self.save_histogram(
            sum(list(self.analysis["stops_by_route"].values()), []),
            f"{save_dir}/stop_by_routes_all.png",
            "stop % all",
            (0, 6000),
        )

        # temporary remove:
        if "all_stop_percentage" not in self.analysis:
            self.analysis["all_stop_percentage"] = defaultdict(lambda: [])
            for episode_vs in self.analysis["vehicle_states"]:
                episode_stop_steps = defaultdict(lambda: [])
                for vs in episode_vs.values():
                    episode_stop_steps[vs["route"]].append(vs["stop_step"])
                for k, v in episode_stop_steps.items():
                    without_zeros = [e for e in v if e > 0]
                    self.analysis["all_stop_percentage"][k].append(
                        len(without_zeros) / len(v)
                    )

        for v_r in self.analysis["all_stop_percentage"].keys():
            self.save_histogram(
                self.analysis["all_stop_percentage"][v_r],
                f"{save_dir}/stop_percentage{v_r}.png",
                f"stop percentage {v_r}",
                (0, 1),
            )

    def generate_figures2(self, save_dir, failure_time, num_bins=20):
        print("Generating Analysis 2")
        total_travel_time_list = []
        travel_percentile_list = []
        blocked_travel_time_list = []
        num_vehicle_list = []
        statistics_list = []
        failed_list = []
        total_south_west = 0
        solved_south_west = 0
        num_scenarios = len(self.all_simulation_data)

        for scenario in self.all_simulation_data:
            statistics = {"destination": {}, "route": {}}
            scenario_travel_time = []
            scenario_blocked_time = []
            for veh_id in scenario["vehicle_data"]:
                destination_key = scenario["vehicle_data"][veh_id]["destination"]
                route_key = scenario["vehicle_data"][veh_id]["route"]

                is_successful = scenario["vehicle_data"][veh_id]["success"]
                is_rerouted = not scenario["vehicle_data"][veh_id]["rerouted"] is None
                is_still_running = not is_successful and not is_rerouted
                travel_time = scenario["vehicle_data"][veh_id]["travel_time"]
                blocked_time = scenario["vehicle_data"][veh_id]["blocked_time"]
                has_failed = travel_time > failure_time or is_rerouted

                _from, _to = route_key.split("|")
                _from, _to = _from.split("-")[1], _to.split("-")[1]
                if _from == "south" and _to == "west":
                    total_south_west += 1
                    if travel_time < 101:
                        solved_south_west += 1

                if not destination_key in statistics["destination"]:
                    statistics["destination"][destination_key] = {
                        "blocked_time": [],
                        "travel_time": [],
                        "success": 0,
                        "rerouted": 0,
                        "in_simulation": 0,
                        "failure_time": 0,
                        "num_vehicles": 0,
                    }
                if not route_key in statistics["route"]:
                    statistics["route"][route_key] = {
                        "blocked_time": [],
                        "travel_time": [],
                        "success": 0,
                        "rerouted": 0,
                        "in_simulation": 0,
                        "failure_time": 0,
                        "num_vehicles": 0,
                    }

                statistics["destination"][destination_key]["success"] += int(
                    is_successful
                )
                statistics["destination"][destination_key]["rerouted"] += int(
                    is_rerouted
                )
                statistics["destination"][destination_key]["in_simulation"] += int(
                    is_still_running
                )
                statistics["destination"][destination_key]["failure_time"] += int(
                    has_failed
                )
                statistics["destination"][destination_key]["num_vehicles"] += 1

                statistics["destination"][destination_key]["travel_time"].append(
                    travel_time
                )
                statistics["destination"][destination_key]["blocked_time"].append(
                    blocked_time
                )

                statistics["route"][route_key]["success"] += int(is_successful)
                statistics["route"][route_key]["rerouted"] += int(is_rerouted)
                statistics["route"][route_key]["in_simulation"] += int(is_still_running)
                statistics["route"][route_key]["failure_time"] += int(has_failed)
                statistics["route"][route_key]["num_vehicles"] += 1

                statistics["route"][route_key]["travel_time"].append(travel_time)
                statistics["route"][route_key]["blocked_time"].append(blocked_time)

            travel_time_list = []
            blocked_time_list = []
            scenario_failed = 0
            for destination_key in statistics["destination"]:
                travel_time_list.extend(
                    statistics["destination"][destination_key]["travel_time"]
                )
                blocked_time_list.extend(
                    statistics["destination"][destination_key]["blocked_time"]
                )
                scenario_failed += statistics["destination"][destination_key][
                    "failure_time"
                ]

            scenario_99_percentile_travel_time = np.percentile(travel_time_list, 99)
            scenario_total_travel_time = np.sum(travel_time_list)
            scenario_blocked_time = np.sum(blocked_time_list)

            total_travel_time_list.append(scenario_total_travel_time)
            travel_percentile_list.append(scenario_99_percentile_travel_time)
            blocked_travel_time_list.append(scenario_blocked_time)

            failed_list.append(scenario_failed / max(len(scenario["vehicle_data"]), 1))
            num_vehicle_list.append(len(scenario["vehicle_data"]))
            statistics_list.append(statistics)

        # Append all scenario statistics together
        all_statistics = {"destination": {}, "route": {}}
        for scenario_statistics in statistics_list:
            for destination_key in scenario_statistics["destination"]:
                if not destination_key in all_statistics["destination"]:
                    all_statistics["destination"][destination_key] = {
                        "blocked_time": [],
                        "travel_time": [],
                        "success": [],
                        "rerouted": [],
                        "in_simulation": [],
                        "failure_time": [],
                        "num_vehicles": [],
                    }
                all_statistics["destination"][destination_key]["success"].append(
                    scenario_statistics["destination"][destination_key]["success"]
                )
                all_statistics["destination"][destination_key]["rerouted"].append(
                    scenario_statistics["destination"][destination_key]["rerouted"]
                )
                all_statistics["destination"][destination_key]["in_simulation"].append(
                    scenario_statistics["destination"][destination_key]["in_simulation"]
                )
                all_statistics["destination"][destination_key]["travel_time"].extend(
                    scenario_statistics["destination"][destination_key]["travel_time"]
                )
                all_statistics["destination"][destination_key]["blocked_time"].extend(
                    scenario_statistics["destination"][destination_key]["blocked_time"]
                )
                all_statistics["destination"][destination_key]["failure_time"].append(
                    scenario_statistics["destination"][destination_key]["failure_time"]
                    / max(
                        scenario_statistics["destination"][destination_key][
                            "num_vehicles"
                        ],
                        1,
                    )
                )
                all_statistics["destination"][destination_key]["num_vehicles"].append(
                    scenario_statistics["destination"][destination_key]["num_vehicles"]
                )

            for route_key in scenario_statistics["route"]:
                if not route_key in all_statistics["route"]:
                    all_statistics["route"][route_key] = {
                        "blocked_time": [],
                        "travel_time": [],
                        "success": [],
                        "rerouted": [],
                        "in_simulation": [],
                        "failure_time": [],
                        "num_vehicles": [],
                    }
                all_statistics["route"][route_key]["success"].append(
                    scenario_statistics["route"][route_key]["success"]
                )
                all_statistics["route"][route_key]["rerouted"].append(
                    scenario_statistics["route"][route_key]["rerouted"]
                )
                all_statistics["route"][route_key]["in_simulation"].append(
                    scenario_statistics["route"][route_key]["in_simulation"]
                )
                all_statistics["route"][route_key]["travel_time"].extend(
                    scenario_statistics["route"][route_key]["travel_time"]
                )
                all_statistics["route"][route_key]["blocked_time"].extend(
                    scenario_statistics["route"][route_key]["blocked_time"]
                )
                all_statistics["route"][route_key]["failure_time"].append(
                    scenario_statistics["route"][route_key]["failure_time"]
                    / max(scenario_statistics["route"][route_key]["num_vehicles"], 1)
                )
                all_statistics["route"][route_key]["num_vehicles"].append(
                    scenario_statistics["route"][route_key]["num_vehicles"]
                )

        # For each destination and route, we generate 5 histogram ....  + 2 for the travel time and percentile ...
        os.makedirs(save_dir, exist_ok=True)
        for destination_key in all_statistics["destination"]:
            sub_figure_dir = os.path.join(save_dir, destination_key)
            os.makedirs(sub_figure_dir, exist_ok=True)
            for statistics_key in all_statistics["destination"][destination_key]:
                self.save_histogram(
                    all_statistics["destination"][destination_key][statistics_key],
                    f"{sub_figure_dir}/{statistics_key}.png",
                    statistics_key,
                )
        for route_key in all_statistics["route"]:
            sub_figure_dir = os.path.join(save_dir, route_key)
            os.makedirs(sub_figure_dir, exist_ok=True)
            for statistics_key in all_statistics["route"][route_key]:
                self.save_histogram(
                    all_statistics["route"][route_key][statistics_key],
                    f"{sub_figure_dir}/{statistics_key}.png",
                    statistics_key,
                )

        self.save_histogram(
            total_travel_time_list,
            f"{save_dir}/total_travel_time.png",
            "total_travel_time",
        )

        self.save_histogram(
            travel_percentile_list,
            f"{save_dir}/99_percentile_travel_time.png",
            "99_percentile_travel_time",
        )

        self.save_histogram(
            blocked_travel_time_list,
            f"{save_dir}/blocked_travel_time.png",
            "blocked_travel_time",
        )

        self.save_histogram(
            num_vehicle_list, f"{save_dir}/num_vehicles.png", "num_vehicles",
        )

        self.save_histogram(
            failed_list, f"{save_dir}/failed_vehicles.png", "failed_vehicles",
        )

        print("Solved south-west:{:.2f}".format(solved_south_west / total_south_west))

    def save_data(self, save_dir):
        with open(f"{save_dir}/analysis.pkl", "wb") as handle:
            dill.dump([self.analysis, self.all_simulation_data], handle)
        print(f"saved data to {save_dir}/analysis.pkl")

    def load_data(self, path):
        with open(path, "rb") as handle:
            self.analysis, self.all_simulation_data = dill.load(handle)
        print(f"loaded data {len(self.analysis)} from {path}")

    def run(self, **params):
        super().run(**params, analyze_func=self.analyze)
