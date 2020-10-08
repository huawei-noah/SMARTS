from zoo.evaluation.metrics.utils import (
    get_scenario_agents_data,
    save_evaluation_result_to_json,
)


class KinematicsEvaluation:
    def __init__(self, scenarios_data_path, result_path, agent_name_list):
        self.scenarios_data_path = scenarios_data_path
        self.result_path = result_path
        self.agent_name_list = agent_name_list

    def run_evaluation(self):
        scenario_agents_data = get_scenario_agents_data(
            self.scenarios_data_path, self.agent_name_list
        )
        kinematics_score = {}
        for scenario_name, data in scenario_agents_data.items():
            agent_result = self.kinematics_result(data)
            kinematics_score[scenario_name] = agent_result
        save_evaluation_result_to_json(kinematics_score, self.result_path, "kinematics")

    def kinematics_result(self, json_curve_list):
        pos_curve_list = []
        speed_curve_list = []
        max_acceleration = 8
        max_speed = 34
        for json_data in json_curve_list:
            pos_curve_list.append(json_data["cartesian_pos_list"])
            speed_curve_list.append(json_data["speed_list"])
        kinematics_result_dict = {}
        for index, agent_name in enumerate(self.agent_name_list):
            agent_speed = speed_curve_list[index]
            agent_acceleration = [
                (agent_speed[i + 1] - agent_speed[i]) / 0.1
                for i in range(len(agent_speed) - 1)
            ]
            temp_result_list = []
            for step, speed in enumerate(agent_speed):
                if step == 0 and abs(speed) > max_speed:
                    temp_result_list.append(step)
                elif step != 0 and abs(speed) > max_speed >= abs(agent_speed[step - 1]):
                    temp_result_list.append(step)
            for step, accel in enumerate(agent_acceleration):
                if step == 0 and abs(accel) > max_acceleration:
                    temp_result_list.append(step)
                elif step != 0 and abs(accel) > max_acceleration >= abs(
                    agent_acceleration[step - 1]
                ):
                    temp_result_list.append(step)
            kinematics_result_dict[agent_name] = sorted(list(set(temp_result_list)))
        return kinematics_result_dict
