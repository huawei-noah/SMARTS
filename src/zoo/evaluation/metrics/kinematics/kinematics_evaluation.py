import json
import os

import numpy as np
import xlrd
import xlwt
from numpy.linalg import det
from scipy.signal import savgol_filter
from xlutils import copy

from zoo.evaluation.metrics.utils import (
    map_agent_to_json_file,
    save_evaluation_result_to_json,
)


class KinematicsEvaluation:
    def __init__(self, scenarios_data_path, result_path, agent_name_list):
        self.scenarios_data_path = scenarios_data_path
        self.result_path = result_path
        self.agent_name_list = agent_name_list
        self.vehicle_width = 1.8
        self.gravity_height = 0.5
        self.data_file = os.path.join(self.result_path, "central_kinematics_data.xls")
        self.sliding_step = 10
        self.max_radius = 100
        self.max_speed = 42
        self.mini_radius = 8
        self.mini_value = 1e-3

    def run_evaluation(self):
        kinematics_score = self.kinematics_algorithm()
        save_evaluation_result_to_json(kinematics_score, self.result_path, "kinematics")

    def kinematics_algorithm(self):
        scenarios_name_list = sorted(os.listdir(self.scenarios_data_path))
        scenarios_path_list = [
            os.path.join(self.scenarios_data_path, s_p) for s_p in scenarios_name_list
        ]
        scenario_agents_data = dict.fromkeys(scenarios_name_list)
        scenario_npcs_data = dict.fromkeys(scenarios_name_list)

        for index, scenario_path in enumerate(scenarios_path_list):
            scenario_name = scenarios_name_list[index]

            all_json_files = [
                os.path.join(scenario_path, js) for js in os.listdir(scenario_path)
            ]
            agents_data = []
            npcs_data = []
            json_files = []
            for json_file in all_json_files:
                if os.path.isfile(json_file):
                    json_files.append(json_file)
            json_files_dict = map_agent_to_json_file(self.agent_name_list, json_files)
            for agent_name in self.agent_name_list:
                with open(json_files_dict[agent_name], "r") as f:
                    json_result = json.load(f)
                    agents_data.append(json_result["agent"])
            scenario_agents_data[scenario_name] = agents_data
            scenario_npcs_data[scenario_name] = npcs_data

        result_dict = {}
        for scenario_name, data in scenario_agents_data.items():
            agent_result = self.evaluation_algorithm(data, scenario_name)
            result_dict[scenario_name] = agent_result
        return result_dict

    def evaluation_algorithm(self, json_curve_list, scenario_name):
        pos_curve_list = []
        speed_curve_list = []

        for json_data in json_curve_list:
            pos_curve_list.append(json_data["cartesian_pos_list"])
            speed_curve_list.append(json_data["speed_list"])
        kinematics_result_dict = {}
        for index, agent_name in enumerate(self.agent_name_list):
            agent_speed = speed_curve_list[index]
            agent_pos = pos_curve_list[index]
            agent_acceleration = [
                (agent_speed[i + 1] - agent_speed[i]) / 0.1
                for i in range(len(agent_speed) - 1)
            ]
            temp_result_list = []
            pos_list = [[round(val, 2) for val in pos] for pos in agent_pos]
            temp_result_list.extend(
                self.rollover_detection(
                    pos_list, agent_speed, scenario_name, agent_name
                )
            )
            temp_result_list.extend(self.speed_detection(agent_speed))
            temp_result_list.extend(self.acceleration_detection(agent_acceleration))
            kinematics_result_dict[agent_name] = sorted(list(set(temp_result_list)))
        return kinematics_result_dict

    def rollover_detection(self, pos_list, speed_list, scenario_name, agent_name):
        result_list = []
        pos_x, pos_y = [i[0] for i in pos_list], [i[1] for i in pos_list]
        x_smooth, y_smooth = self.savitzky_smooth(pos_x), self.savitzky_smooth(pos_y)
        x_smooth = [round(i, 2) for i in x_smooth]
        y_smooth = [round(i, 2) for i in y_smooth]
        road_point = np.array(
            [[x_smooth[i], y_smooth[i]] for i in range(len(x_smooth))]
        )
        curvature_list = self.get_curvature(road_point)
        radius_list = self.get_radius(curvature_list)
        max_speed_list = self.get_max_speed(radius_list)

        for step, curvature in enumerate(curvature_list):
            radius, max_speed = radius_list[step], max_speed_list[step]
            if max_speed < speed_list[step] or radius < self.mini_radius:
                if step and (step - 1) not in result_list:
                    result_list.append(step)
                else:
                    result_list.append(step)

        return result_list

    def speed_detection(self, speed_list):
        result_list = []
        max_speed = 34
        for step, speed in enumerate(speed_list):
            if step == 0 and abs(speed) > max_speed:
                result_list.append(step)
            elif step != 0 and abs(speed) > max_speed >= abs(speed_list[step - 1]):
                result_list.append(step)
        return result_list

    def acceleration_detection(self, acceleration_list):
        result_list = []
        max_acceleration = 8
        for step, accel in enumerate(acceleration_list):
            if step == 0 and abs(accel) > max_acceleration:
                result_list.append(step)
            elif step != 0 and abs(accel) > max_acceleration >= abs(
                acceleration_list[step - 1]
            ):
                result_list.append(step)
        return result_list

    def get_curvature(self, road_points):
        dx_dt = np.gradient(road_points[:, 0])
        dy_dt = np.gradient(road_points[:, 1])
        d2x_dt2 = np.gradient(dx_dt)
        d2y_dt2 = np.gradient(dy_dt)
        curvature_k = (d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (
            dx_dt * dx_dt + dy_dt * dy_dt
        ) ** 1.5
        curvature_list = curvature_k.tolist()

        return curvature_list

    def get_radius(self, curvature_list):
        radius_list = []
        for i in curvature_list:
            if abs(i) > self.mini_value:
                radius = abs(1 / i)
                radius = radius if radius < self.max_radius else self.max_radius
            else:
                radius = self.max_radius
            radius_list.append(radius)

        return radius_list

    def get_max_speed(self, radius_list):
        max_speed_list = []
        for r in radius_list:
            max_speed = pow((self.vehicle_width * 9.8 * r / self.gravity_height), 0.5)
            max_speed = max_speed if max_speed <= self.max_speed else self.max_speed
            max_speed_list.append(max_speed)

        return max_speed_list

    def savitzky_smooth(self, y_origin):
        y_smooth = savgol_filter(y_origin, 21, 3, mode="nearest")
        return y_smooth
