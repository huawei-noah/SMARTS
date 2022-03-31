import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from zoo.evaluation.metrics.diversity.utils import eval_diversity
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

    def diversity_data_analyze(self, agent_list):
        scenarios_name_list = sorted(os.listdir(self.scenarios_data_path))
        scenarios_path_list = [
            os.path.join(self.scenarios_data_path, s_p) for s_p in scenarios_name_list
        ]
        scenario_score_data = dict.fromkeys(scenarios_name_list)
        for index, scenario_path in enumerate(scenarios_path_list):
            scenario_name = scenarios_name_list[index]
            agent_score_data = dict.fromkeys(agent_list)
            json_files = list(Path(scenario_path).glob("**/*json"))
            json_file_dict = map_agent_to_json_file(
                self.all_agent_name_list, json_files
            )
            for agent_name in agent_list:
                with json_file_dict[agent_name].open() as f:
                    json_result = json.load(f)
                    if not json_result.get("npc"):
                        # There is no data of NPC vehicles in the scenario
                        return
                    data_ego = json_result.get("npc")[0]
                    data_agent = json_result.get("agent")
                    pos_ego = np.array(data_ego.get("cartesian_pos_list"))[:, 0:2]
                    pos_agent = np.array(data_agent.get("cartesian_pos_list"))[:, 0:2]
                    time_ego = np.array(data_ego.get("time_list"))
                    time_agent = np.array(data_agent.get("time_list"))
                    speed_ego = np.array(data_ego.get("speed_list"))
                    speed_agent = np.array(data_agent.get("speed_list"))
                    time_score, dist_score = eval_diversity(
                        pos_ego, pos_agent, speed_ego, speed_agent, time_ego, time_agent
                    )
                    agent_score_data[agent_name] = [time_score, dist_score]
                    agent_score_data[agent_name] = [time_score, dist_score]
                    scenario_score_data[scenario_name] = agent_score_data
        return scenario_score_data

    def run_evaluation(self):
        result_file = os.path.join(
            self.csv_file_result_path, "diversity_evaluation_result.csv"
        )
        for actor_name, agent_list in self.agent_groups.items():
            result_dict = self.diversity_data_analyze(agent_list)
            # pytype: disable=attribute-error
            scenario_name_list = list(result_dict.keys())
            df = pd.DataFrame(
                {
                    "scenario": scenario_name_list,
                }
            )
            df.set_index(["scenario"], inplace=True)
            for agent in agent_list:
                for scenario_name, result_data in result_dict.items():
                    # pytype: enable=attribute-error
                    df.loc[scenario_name, "%s-time_score" % agent] = result_data[agent][
                        0
                    ]
                    df.loc[scenario_name, "%s-dis_score" % agent] = result_data[agent][
                        1
                    ]
                    df.loc[""] = ""
                write_csv_file(result_file, actor_name)
                df.to_csv(result_file, mode="a")

        for actor_name, agent_list in self.agent_groups.items():
            write_csv_file(result_file, actor_name)
            for agent in agent_list:
                write_csv_file(result_file, agent)
