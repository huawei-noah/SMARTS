from collections import defaultdict

from zoo.evaluation.metrics.utils import (
    get_scenario_agents_data,
    save_evaluation_result_to_json,
)


class CollisionEvaluation:
    def __init__(self, scenarios_data_path, result_path, agent_name_list):
        self.scenarios_data_path = scenarios_data_path
        self.result_path = result_path
        self.agent_name_list = agent_name_list

    def run_evaluation(self):
        scenario_agents_data = get_scenario_agents_data(
            self.scenarios_data_path, self.agent_name_list
        )
        collision_score = {}
        for scenario_name, data in scenario_agents_data.items():
            agent_result = self.collision_result(data)
            collision_score[scenario_name] = agent_result
        save_evaluation_result_to_json(collision_score, self.result_path, "collision")

    def collision_result(self, json_curve_list):
        collision_data_list = []
        collision_result_dict = defaultdict(list)
        for json_data in json_curve_list:
            collision_data_list.append(json_data["collision"])
        for i, collision_list in enumerate(collision_data_list):
            for index, collision_result in enumerate(collision_list):
                if index:
                    if int(collision_result) == 1 and collision_list[index - 1] != 1:
                        collision_result_dict[self.agent_name_list[i]].append(index)
                else:
                    if int(collision_result) == 1:
                        collision_result_dict[self.agent_name_list[i]].append(index)
        return collision_result_dict
