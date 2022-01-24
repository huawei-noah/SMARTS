import argparse
import collections
import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from matplotlib import pyplot as plt

"""
Example usage,

$ python tools/evalution.py ./data_replay/1597183388
"""


def extract_data(json_file, step_num, timestep):
    time = 0
    all_data = collections.OrderedDict()
    vehicle_data = {}
    with open(json_file, "r") as f:
        step = 0
        for line in f:
            if step > step_num:
                break
            step += 1

            data = json.loads(line.rstrip("\n"))
            for vehicle_id, state in data["traffic"].items():
                if vehicle_id not in vehicle_data:
                    vehicle_data[vehicle_id] = defaultdict(list)

                vehicle_data[vehicle_id]["speed_list"].append(state["speed"])
                vehicle_data[vehicle_id]["cartesian_pos_list"].append(state["position"])
                vehicle_data[vehicle_id]["time_list"].append(time)
                if state["events"]:
                    vehicle_data[vehicle_id]["collision"].append(
                        bool(state["events"]["collisions"])
                    )
                    vehicle_data[vehicle_id]["off_road"].append(
                        bool(state["events"]["off_road"])
                    )
                    time += timestep

    npc_data = [
        data for vehicle_id, data in vehicle_data.items() if "npc" in vehicle_id
    ]
    social_agent_data = [
        data for vehicle_id, data in vehicle_data.items() if "npc" not in vehicle_id
    ]
    assert len(social_agent_data) == 1

    all_data["npc"] = npc_data
    all_data["agent"] = social_agent_data[0]

    return all_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser("policy-evaluation")
    parser.add_argument(
        "path",
        help="Directory path containing data files (.jsonl)",
        type=str,
    )

    parser.add_argument(
        "result_path",
        help="Directory path containing data files",
        type=str,
    )

    parser.add_argument(
        "agent_name",
        help="Agent name",
        type=str,
    )

    parser.add_argument("--step-num", help="Number of steps", type=str, default=600)

    parser.add_argument(
        "--timestep-sec",
        type=float,
        default=0.1,
        help="Timestep, can be seen as the pause duration between frames",
    )

    args = parser.parse_args()
    agent_name = args.agent_name
    jsonl_paths = list(Path(args.path).glob("*.jsonl"))
    step_num = int(args.step_num.split(":")[-1])

    assert len(jsonl_paths) == 1
    for jsonl in jsonl_paths:
        data = extract_data(jsonl, step_num, args.timestep_sec)
        time_suffix = datetime.now().strftime("%Y%m%d-%H%M%S")
        scenario_path = Path(args.path).parent.parent
        result_path = Path(args.result_path)
        result_file = (
            result_path
            / f"evaluation-data_{scenario_path.name}_{agent_name}_{time_suffix}.json"
        )
        with open(result_file, "w") as f:
            json.dump(data, f)
