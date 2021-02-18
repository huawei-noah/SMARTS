import csv
import json
import os
from pathlib import Path


def get_evaluation_result(_data, evaluation_result, evaluation_type):
    for agent_name, value in _data.items():
        if value:
            if "pass" in evaluation_result and agent_name in evaluation_result["pass"]:
                evaluation_result["pass"].remove(agent_name)
            if agent_name in evaluation_result["fail"]:
                evaluation_result["fail"][agent_name][evaluation_type] = value
            else:
                evaluation_result["fail"][agent_name] = {}
                evaluation_result["fail"][agent_name][evaluation_type] = value
        else:
            if agent_name not in evaluation_result["fail"]:
                if agent_name not in evaluation_result["pass"]:
                    evaluation_result["pass"].append(agent_name)
                else:
                    evaluation_result["pass"] = [agent_name]
    return evaluation_result


def map_agent_to_json_file(agent_names, all_json_file):
    json_file_dict = {}
    for json_file in all_json_file:
        file_name = os.path.basename(json_file)
        candidate_agents = [name for name in agent_names if name in file_name]
        most_likely_agent_name = max(candidate_agents, key=lambda name: len(name))

        assert most_likely_agent_name not in json_file_dict
        json_file_dict[most_likely_agent_name] = json_file
    return json_file_dict


def save_evaluation_result_to_json(result_datas, result_path, evaluation_type):
    for scenario_name, data in result_datas.items():
        scenario_path = os.path.join(result_path + "/" + scenario_name)
        json_result_file = os.path.join(scenario_path, "evaluation_results.json")
        if not os.path.exists(json_result_file):
            result_data = get_evaluation_result(
                data, {"pass": [], "fail": {}}, evaluation_type
            )
        else:
            with open(json_result_file, "r") as f:
                json_result = json.load(f)
            result_data = get_evaluation_result(data, json_result, evaluation_type)
        with open(json_result_file, "w") as f:
            json.dump(result_data, f)


def get_scenario_agents_data(scenarios_data_path, agent_name_list):
    scenarios_name_list = sorted(os.listdir(scenarios_data_path))
    scenarios_path_list = [
        os.path.join(scenarios_data_path, s_p) for s_p in scenarios_name_list
    ]
    scenario_agents_data = dict.fromkeys(scenarios_name_list)

    for index, scenario_path in enumerate(scenarios_path_list):
        scenario_name = scenarios_name_list[index]
        json_files = list(Path(scenario_path).glob("**/*json"))
        agents_data = []
        json_files_dict = map_agent_to_json_file(agent_name_list, json_files)
        for agent_name in agent_name_list:
            with open(json_files_dict[agent_name], "r") as f:
                json_result = json.load(f)
                agents_data.append(json_result["agent"])
        scenario_agents_data[scenario_name] = agents_data
    return scenario_agents_data


def write_csv_file(result, content):
    try:
        with open(result, "a") as csv_file:
            writer = csv.writer(csv_file, dialect="excel")
            writer.writerow([content])
    except Exception as e:
        print("Write an csv file to path: %s, Case: %s" % (result, e))
