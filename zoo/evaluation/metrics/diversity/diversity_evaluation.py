import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from zoo.evaluation.metrics.utils import map_agent_to_json_file, write_csv_file


class DiversityEvaluation:
    def __init__(
        self,
        scenarios_data_path,
        csv_file_result_path,
        agent_groups,
        all_agent_name_list,
    ):
        self.scenarios_data_path = scenarios_data_path
        self.csv_file_result_path = csv_file_result_path
        self.agent_groups = agent_groups
        self.all_agent_name_list = all_agent_name_list
        self.deviation_data = {}

    def run_evaluation(self):
        result_file = os.path.join(
            self.csv_file_result_path, "diversity_evaluation_result.csv"
        )
        for actor_name, agent_list in self.agent_groups.items():
            result_dict = self.diversity_result(agent_list)
            scenario_name_list = list(result_dict.keys())
            empty_list = [""] * len(scenario_name_list)
            df = pd.DataFrame(
                {
                    "scenario": scenario_name_list,
                    "agent_score": empty_list,
                    "agent_result": empty_list,
                }
            )
            df.set_index(["scenario"], inplace=True)
            for scenario_name, result_data in result_dict.items():
                df.loc[scenario_name, "agent_score"] = result_data[0]
                if result_data[0] >= 0.8:
                    df.loc[scenario_name, "agent_result"] = "pass"
                else:
                    df.loc[scenario_name, "agent_result"] = "fail"
            df.loc[""] = ""
            write_csv_file(result_file, actor_name)
            df.to_csv(result_file, mode="a")
            self.criterion_curve_plan_plot(actor_name, agent_list)

        for actor_name, agent_list in self.agent_groups.items():
            write_csv_file(result_file, actor_name)
            for agent in agent_list:
                write_csv_file(result_file, agent)
        write_csv_file(result_file, "")
        write_csv_file(
            result_file, "Agent above 0.8 is good, while agent below 0.8 is bad"
        )
        write_csv_file(result_file, "NPC below 0.1 is good, while NPC above 0.1 is bad")

    def diversity_result(self, agent_list):
        scenarios_name_list = sorted(os.listdir(self.scenarios_data_path))
        scenarios_path_list = [
            os.path.join(self.scenarios_data_path, s_p) for s_p in scenarios_name_list
        ]
        scenario_agents_data = dict.fromkeys(scenarios_name_list)
        scenario_npcs_data = dict.fromkeys(scenarios_name_list)
        for index, scenario_path in enumerate(scenarios_path_list):
            scenario_name = scenarios_name_list[index]
            json_files = list(Path(scenario_path).glob("**/*json"))
            agents_data = []
            npcs_data = []
            json_file_dict = map_agent_to_json_file(
                self.all_agent_name_list, json_files
            )
            for agent_name in agent_list:
                with json_file_dict[agent_name].open() as f:
                    json_result = json.load(f)
                    agents_data.append(json_result["agent"])
                    if json_result["npc"]:
                        npcs_data.append(json_result["npc"][0])
            scenario_agents_data[scenario_name] = agents_data
            scenario_npcs_data[scenario_name] = npcs_data
        test_evaluation_params = [
            {
                "distance_threshold": 3,
                "speed_threshold": 1,
                "exceed_ratio_threshold": 0.2,
            }
        ]
        result_dict = dict.fromkeys(scenarios_name_list)
        for param in test_evaluation_params:
            for (scenario_name, data) in scenario_agents_data.items():
                agents_result = self.diversity_algorithm(
                    scenario_name, data, "agent", param
                )
                result_dict[scenario_name] = [agents_result[0]]
            for (scenario_name, data) in scenario_npcs_data.items():
                if data:
                    agents_result = self.diversity_algorithm(
                        scenario_name, data, "npc", param
                    )
                    result_dict[scenario_name].append(agents_result[0])
                else:
                    result_dict[scenario_name].append(-1)
        return result_dict

    def diversity_algorithm(
        self,
        scenario_name,
        json_curve_list,
        agent_type,
        evaluation_params={
            "distance_threshold": 3.5,
            "speed_threshold": 5,
            "exceed_ratio_threshold": 0.15,
        },
    ):
        if not evaluation_params:
            evaluation_params = {
                "distance_threshold": 3.5,
                "speed_threshold": 5,
                "exceed_ratio_threshold": 0.15,
            }

        # preprocess
        pos_curve_list = []
        speed_curve_list = []
        pos_distance_list = []
        speed_distance_list = []

        for json_data in json_curve_list:
            pos_curve_list.append(json_data["cartesian_pos_list"])
            speed_curve_list.append(json_data["speed_list"])
        criterion_pos_curve = self.get_criterion_pos_curve(pos_curve_list)
        criterion_speed_curve = self.get_criterion_speed_curve(speed_curve_list)
        for index in range(len(pos_curve_list)):
            pos_distance_list.append(
                self.cal_pos_curve_distance(criterion_pos_curve, pos_curve_list[index])
            )
        for index in range(len(speed_curve_list)):
            speed_distance_list.append(
                self.cal_speed_curve_distance(
                    criterion_speed_curve, speed_curve_list[index]
                )
            )
        if agent_type == "agent":
            self.deviation_data[scenario_name] = {
                "cal_pos_curve_list": pos_distance_list,
                "cal_speed_curve_list": speed_distance_list,
            }

        speed_absolute_distance_list = np.fabs(speed_distance_list)
        distance_threshold = evaluation_params["distance_threshold"]  # 3.5
        speed_threshold = evaluation_params["speed_threshold"]  # 5
        exceed_ratio_threshold = evaluation_params["exceed_ratio_threshold"]  # 0.15

        dis_exceed_number_ratio = np.array(
            [dl[dl > distance_threshold].size / dl.size for dl in pos_distance_list]
        ).mean()
        speed_exceed_number_ratio = np.array(
            [
                sl[sl > speed_threshold].size / sl.size
                for sl in speed_absolute_distance_list
            ]
        ).mean()
        exceed_ratio_score = (
            dis_exceed_number_ratio
            if dis_exceed_number_ratio > speed_exceed_number_ratio
            else speed_exceed_number_ratio
        ) / exceed_ratio_threshold
        exceed_ratio_score = exceed_ratio_score if exceed_ratio_score < 1 else 1
        dis_even_deviation_value_ratio = (
            np.asarray(pos_distance_list).mean() / distance_threshold
        )
        speed_even_deviation_value_ratio = (
            np.asarray(speed_absolute_distance_list).mean() / speed_threshold
        )
        even_score = (
            dis_even_deviation_value_ratio
            if dis_even_deviation_value_ratio > speed_even_deviation_value_ratio
            else speed_even_deviation_value_ratio
        )
        even_score = even_score if even_score < 1 else 1
        diversity_score = max([exceed_ratio_score, even_score])

        return [
            diversity_score,
            exceed_ratio_score,
            even_score,
            dis_exceed_number_ratio,
            speed_exceed_number_ratio,
            dis_even_deviation_value_ratio,
            speed_even_deviation_value_ratio,
        ]

    def get_criterion_pos_curve(self, pos_curve_list):
        criterion_curve = []
        curves_num = len(pos_curve_list)
        min_curve_list = pos_curve_list[0]
        for current_curve in pos_curve_list:
            if len(min_curve_list) >= len(current_curve):
                min_curve_list = current_curve

        for index in range(len(min_curve_list)):
            all_point_x = 0
            all_point_y = 0
            for current_num in range(curves_num):
                all_point_x += pos_curve_list[current_num][index][0]
                all_point_y += pos_curve_list[current_num][index][1]
            criterion_curve.append([all_point_x / curves_num, all_point_y / curves_num])
        return criterion_curve

    def get_criterion_speed_curve(self, speed_curve_list):
        criterion_curve = []
        curves_num = len(speed_curve_list)
        min_curve_list = speed_curve_list[0]
        for current_curve in speed_curve_list:
            if len(min_curve_list) >= len(current_curve):
                min_curve_list = current_curve

        for index in range(len(min_curve_list)):
            all_point = 0
            for current_num in range(curves_num):
                all_point += speed_curve_list[current_num][index]
            criterion_curve.append(all_point / curves_num)
        return criterion_curve

    def cal_pos_curve_distance(self, curve1, curve2):
        dis_value_list = []
        for index in range(len(curve1)):
            dis_value = pow(
                pow((curve1[index][0] - curve2[index][0]), 2)
                + pow((curve1[index][1] - curve2[index][1]), 2),
                0.5,
            )
            dis_value_list.append(dis_value)
        return np.array(dis_value_list)

    def cal_speed_curve_distance(self, curve1, curve2):
        speed_value_list = []
        for index in range(len(curve1)):
            speed_value = curve1[index] - curve2[index]
            speed_value_list.append(speed_value)
        return np.array(speed_value_list)

    def criterion_curve_plan_plot(self, actor_name, agents_name_lists):
        length = len(self.deviation_data)
        agents_num = len(agents_name_lists)
        color_lists = ["r", "y", "b", "g", "c", "m", "y"]

        fig, ax = plt.subplots(length, ncols=2, figsize=(14, 14))
        fig.subplots_adjust(hspace=0.8)
        fig.suptitle("distance&speed_deviation_curve", fontsize=15)
        axes = ax.flatten()
        for i, (scenario_name, data) in enumerate(self.deviation_data.items()):
            agent_pos_offset_list = data["cal_pos_curve_list"]
            pos_offset_list = []
            for j in range(agents_num):
                pos_offset_list += agent_pos_offset_list[j].tolist()
            max_pos_offset = max(pos_offset_list)
            agent_speed_offset_list = data["cal_speed_curve_list"]
            speed_offset_list = []
            for j in range(agents_num):
                speed_offset_list += agent_speed_offset_list[j].tolist()
            max_speed_offset = max(speed_offset_list)

            _x = [k for k in range(len(agent_speed_offset_list[0].tolist()))]
            for index in range(agents_num):
                pos_y = [k for k in agent_pos_offset_list[index].tolist()]
                speed_y = [k for k in agent_speed_offset_list[index]]
                axes[2 * i].set_title(
                    "%s:distance_deviation" % scenario_name, fontsize=12
                )
                axes[2 * i].set_xlabel("step")
                axes[2 * i].set_ylabel("distance_deviation")
                axes[2 * i].plot(_x, pos_y, ":%s" % color_lists[index])
                axes[2 * i].text(
                    0,
                    max_pos_offset * (0.5 + index * 0.15),
                    "%s" % agents_name_lists[index],
                    fontsize=6,
                    color="%s" % color_lists[index],
                    style="normal",
                )
                axes[2 * i + 1].set_title(
                    "%s:speed_deviation" % scenario_name, fontsize=12
                )
                axes[2 * i + 1].set_xlabel("step")
                axes[2 * i + 1].set_ylabel("speed_deviation")
                axes[2 * i + 1].plot(_x, speed_y, ":%s" % color_lists[index])
                axes[2 * i + 1].text(
                    0,
                    max_speed_offset * (0.5 + index * 0.15),
                    "%s" % agents_name_lists[index],
                    fontsize=6,
                    color="%s" % color_lists[index],
                    style="normal",
                )

        time_suffix = datetime.now().strftime("%Y%m%d-%H%M")
        result_file = os.path.join(
            self.csv_file_result_path,
            "%s_distance&speed_deviation-curve_%s.png" % (actor_name, time_suffix),
        )
        plt.savefig(result_file)
