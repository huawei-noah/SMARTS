import os

import pandas as pd

from zoo.evaluation.metrics.utils import write_csv_file


class RankReport:
    def __init__(self, agent_groups, csv_file_result_path):
        self.csv_file_result_path = csv_file_result_path
        self.weight_ritio = {"collision": 0.3, "offroad": 0.4, "kinematics": 0.3}
        self.weight_ritios = {
            "collision": 0.2,
            "offroad": 0.2,
            "kinematics": 0.2,
            "diversity": 0.4,
        }
        self.result_file = os.path.join(self.csv_file_result_path, "rank.csv")
        self.origin_file = os.path.join(self.csv_file_result_path, "report.csv")
        self.group_agents_list = []
        for group_name, agents_list in agent_groups.items():
            for agent in agents_list:
                self.group_agents_list.append(group_name + ":" + agent)
        self.agents_list = [agent.split(":")[-1] for agent in self.group_agents_list]
        self.group_list = [agent.split(":")[0] for agent in self.group_agents_list]
        self.origin_df = pd.read_csv(self.origin_file, nrows=len(self.agents_list))
        self.val_list = ["collision", "offroad", "kinematics", "diversity"]

    def result_output(self):
        if os.path.isfile(self.result_file):
            os.remove(self.result_file)
        self.origin_df.set_index(["agent"], inplace=True)
        write_csv_file(self.result_file, "ranking without diversity:")
        self.rank_result_to_csv(self.weight_ritio, self.val_list[:-1])
        write_csv_file(self.result_file, "ranking with diversity:")
        self.rank_result_to_csv(self.weight_ritios, self.val_list)

    def rank_result_to_csv(self, weight_ritio, val_list):
        agents_score = {}
        for agent in self.agents_list:
            agent_score = 0
            for value in list(weight_ritio.keys()):
                agent_score += (
                    float((self.origin_df.loc[agent, value])[:-1])
                    * weight_ritio[value]
                    * 0.01
                )
                agents_score[agent] = agent_score
        agents_score = sorted(agents_score.items(), key=lambda x: x[1], reverse=True)
        weight_score_ranking, agent_ranking = [], []
        for i in agents_score:
            agent_ranking.append(i[0])
            weight_score_ranking.append(i[1])
        blank_list = [""] * len(self.agents_list)
        weight_score_ranking = [
            str(round(i * 100, 2)) + "%" for i in weight_score_ranking
        ]
        if "diversity" in val_list:
            rank_df = pd.DataFrame(
                {
                    "ranking": range(1, len(self.agents_list) + 1),
                    "group": blank_list,
                    "agent": agent_ranking,
                    "summary_result": weight_score_ranking,
                    "collision": blank_list,
                    "offroad": blank_list,
                    "kinematics": blank_list,
                    "diversity": blank_list,
                }
            )
        else:
            rank_df = pd.DataFrame(
                {
                    "ranking": range(1, len(self.agents_list) + 1),
                    "group": blank_list,
                    "agent": agent_ranking,
                    "summary_result": weight_score_ranking,
                    "collision": blank_list,
                    "offroad": blank_list,
                    "kinematics": blank_list,
                }
            )
        rank_df.set_index(["agent"], inplace=True)
        for value in val_list:
            for agent in agent_ranking:
                rank_df.loc[agent, value] = self.origin_df.loc[agent, value]
        for index, agent in enumerate(agent_ranking):
            group_name = self.group_list[self.agents_list.index(agent)]
            rank_df.loc[agent, "group"] = group_name
            if (
                index
                and rank_df.loc[agent, "summary_result"]
                == rank_df.loc[agent_ranking[index - 1], "summary_result"]
            ):
                rank_df.loc[agent, "ranking"] = rank_df.loc[
                    agent_ranking[index - 1], "ranking"
                ]
        rank_df.loc[""] = ""
        rank_df.to_csv(self.result_file, mode="a")
        write_csv_file(
            self.result_file, "weight ritio:" + str(weight_ritio).replace(",", "")
        )
        write_csv_file(self.result_file, "")
