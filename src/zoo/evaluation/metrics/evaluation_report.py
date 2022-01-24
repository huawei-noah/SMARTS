import csv
import json
import os
from collections import defaultdict

import pandas as pd

from zoo.evaluation.metrics.utils import write_csv_file


class EvaluationReport:
    def __init__(self, scenarios_list, agents_list, csv_file_result_path):
        self.scenarios_list = scenarios_list
        self.csv_file_result_path = csv_file_result_path
        self.group_agents_list = []
        for value in agents_list.items():
            for agent in value[1]:
                self.group_agents_list.append(value[0] + ":" + agent)
        self.agents_list = [agent.split(":")[-1] for agent in self.group_agents_list]
        self.group_list = [agent.split(":")[0] for agent in self.group_agents_list]

    def result_output(self):
        result_file = os.path.join(self.csv_file_result_path, "report.csv")
        if os.path.isfile(result_file):
            os.remove(result_file)

        empty_list = [""] * len(self.scenarios_list)
        val_list = ["collision", "offroad", "kinematics", "diversity"]
        data_dict = {}
        for agent in self.agents_list:
            temp_dict = {agent: {}}
            for val in val_list:
                temp_dict[agent].update({val: 0})
            data_dict.update(temp_dict)

        diversity_result = self.read_diversity_result()
        for agent in self.agents_list:
            df = pd.DataFrame(
                {
                    "scenario": self.scenarios_list,
                    "collision": empty_list,
                    "offroad": empty_list,
                    "kinematics": empty_list,
                    "diversity": empty_list,
                }
            )
            df.set_index(["scenario"], inplace=True)
            current_group = self.group_list[self.agents_list.index(agent)]
            df.loc[:, "diversity"] = diversity_result[current_group]
            for scenario in self.scenarios_list:
                scenario_path = os.path.join(self.csv_file_result_path, scenario)
                scenario_json_file = os.path.join(
                    scenario_path, "evaluation_results.json"
                )
                with open(scenario_json_file, "r") as f:
                    json_result = json.load(f)
                pass_agents = json_result["pass"]
                fail_agents = list(json_result["fail"].keys())
                if agent in pass_agents:
                    df.loc[scenario, val_list[:-1]] = "pass"
                elif agent in fail_agents:
                    for value in val_list[:-1]:
                        if value in list((json_result["fail"][agent]).keys()):
                            df.loc[scenario, value] = "fail"
                        else:
                            df.loc[scenario, value] = "pass"
            df.loc[""] = ""
            group_agent_name = self.group_agents_list[self.agents_list.index(agent)]
            write_csv_file(result_file, group_agent_name)
            df.to_csv(result_file, mode="a")

            for scenario in self.scenarios_list:
                for val in val_list:
                    if df.loc[scenario, val] == "pass":
                        data_dict[agent][val] += 1
        for agent in self.agents_list:
            for val in val_list:
                data_dict[agent][val] /= len(self.scenarios_list)
                data_dict[agent][val] = str(round(data_dict[agent][val] * 100, 2)) + "%"
        blank_list = [""] * len(self.agents_list)
        df_result = pd.DataFrame(
            {
                "agent": self.agents_list,
                "collision": blank_list,
                "offroad": blank_list,
                "kinematics": blank_list,
                "diversity": blank_list,
            }
        )
        df_result.set_index(["agent"], inplace=True)
        for agent in self.agents_list:
            for val in val_list:
                df_result.loc[agent, val] = data_dict[agent][val]
        df_result.insert(0, "group", self.group_list)
        df_temp = pd.read_csv(result_file)

        df_result.to_csv(result_file)
        write_csv_file(result_file, "")
        write_csv_file(
            result_file,
            "The result represents the scenario pass rate of each agent under different evaluation items",
        )
        write_csv_file(result_file, "")
        df_temp.to_csv(result_file, mode="a", index_label=False)

    def read_diversity_result(self):
        diversity_csv_file = os.path.join(
            self.csv_file_result_path, "diversity", "diversity_evaluation_result.csv"
        )
        csv_file = open(diversity_csv_file, "r")
        reader = csv.reader(csv_file)
        df_diversity = pd.DataFrame(reader, dtype=str)
        diversity_result = defaultdict(list)
        unique_group_list = sorted(
            list(set(self.group_list)), key=self.group_list.index
        )
        for i in range(len(unique_group_list)):
            index = i * (len(self.scenarios_list) + 3) + 2
            diversity_result[unique_group_list[i]] = df_diversity.iloc[
                list(range(index, index + len(self.scenarios_list))), 2
            ].tolist()
        return diversity_result
