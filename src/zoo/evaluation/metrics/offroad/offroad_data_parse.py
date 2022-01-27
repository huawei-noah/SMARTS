from zoo.evaluation.metrics.utils import (
    get_scenario_agents_data,
    save_evaluation_result_to_json,
)


class OffroadEvaluation:
    def __init__(self, scenarios_data_path, result_path, agent_name_list):
        self.scenarios_data_path = scenarios_data_path
        self.result_path = result_path
        self.agent_name_list = agent_name_list

    def run_evaluation(self):
        scenario_agents_data = get_scenario_agents_data(
            self.scenarios_data_path, self.agent_name_list
        )
        offroad_score = {}
        for scenario_name, data in scenario_agents_data.items():
            agent_result = self.offroad_result(data)
            offroad_score[scenario_name] = agent_result
        save_evaluation_result_to_json(offroad_score, self.result_path, "offroad")

    def offroad_result(self, json_curve_list):
        offroad_data_list = []
        offroad_result_dict = {}
        for json_data in json_curve_list:
            offroad_data_list.append(json_data["off_road"])
        for i, offroad_list in enumerate(offroad_data_list):
            offroad_result_dict[self.agent_name_list[i]] = []
            for index, offroad_result in enumerate(offroad_list):
                if index:
                    if int(offroad_result) == 1 and offroad_list[index - 1] != 1:
                        offroad_result_dict[self.agent_name_list[i]].append(index)
                else:
                    if int(offroad_result) == 1:
                        offroad_result_dict[self.agent_name_list[i]].append(index)
        return offroad_result_dict
