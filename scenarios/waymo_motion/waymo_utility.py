# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# type: ignore

# Visualization and prototyping script for the Waymo motion dataset.
import argparse
import os
import shutil
import yaml
from typing import Dict, List, Tuple, Union
from tabulate import tabulate
from pathlib import Path
import matplotlib.pyplot as plt
from waymo_open_dataset.protos import scenario_pb2

from smarts.core.utils.file import read_tfrecord_file


def convert_polyline(polyline) -> Tuple[List[float], List[float]]:
    xs, ys = [], []
    for p in polyline:
        xs.append(p.x)
        ys.append(p.y)
    return xs, ys


def get_map_features_for_scenario(scenario) -> Dict:
    lanes = []
    map_features = {
        "lane": [],
        "road_line": [],
        "road_edge": [],
        "stop_sign": [],
        "crosswalk": [],
        "speed_bump": [],
    }
    for i in range(len(scenario.map_features)):
        map_feature = scenario.map_features[i]
        key = map_feature.WhichOneof("feature_data")
        if key is not None:
            map_features[key].append(getattr(map_feature, key))
            if key == "lane":
                lanes.append((getattr(map_feature, key), map_feature.id))
    # tls_lanes = get_traffic_light_lanes(scenario)
    return map_features, lanes


def edit_scenario_yaml(yaml_path: str, file_path: str, scenario_id: str):
    with open(yaml_path) as f:
        data_spec = yaml.safe_load(f)
    data_spec["trajectory_dataset"]["input_path"] = file_path
    data_spec["trajectory_dataset"]["scenario_id"] = scenario_id

    with open(yaml_path, "w") as f:
        yaml.dump(data_spec, f, default_flow_style=False)


def get_traffic_light_lanes(scenario) -> List[str]:
    num_steps = len(scenario.timestamps_seconds)
    tls_lanes = []
    for i in range(num_steps):
        dynamic_states = scenario.dynamic_map_states[i]
        for j in range(len(dynamic_states.lane_states)):
            lane_state = dynamic_states.lane_states[j]
            tls_lanes.append(lane_state.lane)
    return tls_lanes


def plot_map(map_features):
    lanes = map_features["lane"][:1]
    lane_points = [convert_polyline(lane.polyline) for lane in lanes]
    # lanes = list(filter(lambda lane: max(lane[1]) > 8150, lanes))
    for xs, ys in lane_points:
        plt.plot(xs, ys, linestyle=":", color="gray")
    for road_line in map_features["road_line"]:
        xs, ys = convert_polyline(road_line.polyline)
        if road_line.type in [1, 4, 5]:
            plt.plot(xs, ys, "y--")
        else:
            plt.plot(xs, ys, "y-")
    for road_edge in map_features["road_edge"]:
        xs, ys = convert_polyline(road_edge.polyline)
        plt.plot(xs, ys, "k-")
    # for crosswalk in map_features["crosswalk"]:
    #     xs, ys = convert_polyline(crosswalk.polygon)
    #     plt.plot(xs, ys, 'k--')
    # for speed_bump in map_features["speed_bump"]:
    #     xs, ys = convert_polyline(speed_bump.polygon)
    #     plt.plot(xs, ys, 'k:')
    for stop_sign in map_features["stop_sign"]:
        plt.scatter(
            stop_sign.position.x, stop_sign.position.y, marker="o", c="#ff0000", alpha=1
        )


def plot_lane(lane):
    xs, ys = convert_polyline(lane.polyline)
    plt.plot(xs, ys, linestyle="-", c="gray")
    # plt.scatter(xs, ys, s=12, c="gray")
    # plt.scatter(xs[0], ys[0], s=12, c="red")


def plot_road_line(road_line):
    xs, ys = convert_polyline(road_line.polyline)
    plt.plot(xs, ys, "y-")
    plt.scatter(xs, ys, s=12, c="y")
    # plt.scatter(xs[0], ys[0], s=12, c="red")


def plot_road_edge(road_edge):
    xs, ys = convert_polyline(road_edge.polyline)
    plt.plot(xs, ys, "k-")
    plt.scatter(xs, ys, s=12, c="black")
    # plt.scatter(xs[0], ys[0], s=12, c="red")


def plot(path: str, scenario_id: str):
    # Find scenario from path with given scenario_id
    dataset = read_tfrecord_file(path)
    scenario = None
    for record in dataset:
        parsed_scenario = scenario_pb2.Scenario()
        parsed_scenario.ParseFromString(bytearray(record))
        if parsed_scenario.scenario_id == scenario_id:
            scenario = parsed_scenario
            break
    if scenario is None:
        errmsg = f"Dataset file does not contain scenario with id: {scenario_id}"
        raise ValueError(errmsg)

    # Get data
    map_features = get_map_features_for_scenario(scenario)

    # Plot map
    fig, ax = plt.subplots()
    ax.set_title(f"Scenario {scenario_id}")
    ax.axis("equal")
    plot_map(map_features)

    mng = plt.get_current_fig_manager()
    mng.resize(1000, 1000)
    # mng.resize(*mng.window.maxsize())
    plt.show()


def dump_plots(out_dir: str, path: str) -> List[str]:
    scenarios = []
    dataset = read_tfrecord_file(path)
    for record in dataset:
        scenario = scenario_pb2.Scenario()
        scenario.ParseFromString(bytearray(record))

        scenario_id = scenario.scenario_id
        scenarios.append(scenario_id)
        map_features = get_map_features_for_scenario(scenario)

        fig, ax = plt.subplots()
        ax.set_title(f"Scenario {scenario_id}")
        plot_map(map_features)
        mng = plt.get_current_fig_manager()
        # mng.resize(*mng.window.maxsize())
        w = 1000
        h = 1000
        mng.resize(w, h)
        # plt.show()

        filename = f"scenario-{scenario_id}.png"
        out_path = os.path.join(out_dir, filename)
        fig = plt.gcf()
        # w, h = mng.window.maxsize()
        dpi = 100
        fig.set_size_inches(w / dpi, h / dpi)
        print(f"Saving {out_path}")
        fig.savefig(out_path, dpi=100)
        plt.close("all")
    return scenarios


def get_scenario_dict(tfrecord_file: str) -> List[str]:
    scenario_dict = {}
    dataset = read_tfrecord_file(tfrecord_file)
    for record in dataset:
        scenario = scenario_pb2.Scenario()
        scenario.ParseFromString(bytearray(record))
        scenario_id = scenario.scenario_id
        scenario_dict[scenario_id] = scenario

    return scenario_dict


def parse_tfrecords(tfrecord_path: str):
    scenarios_per_tfrecord = {}
    if os.path.isdir(tfrecord_path):
        for f in os.listdir(tfrecord_path):
            if ".tfrecord" in f and os.path.isfile(os.path.join(tfrecord_path, f)):
                scenario_dict = get_scenario_dict(os.path.join(tfrecord_path, f))
                scenarios_per_tfrecord[f] = scenario_dict
    else:
        scenarios_per_tfrecord[tfrecord_path] = get_scenario_dict(tfrecord_path)
    return scenarios_per_tfrecord


def display_scenarios_in_tfrecord(tfrecord_path, scenario_dict) -> List[str]:
    scenario_data_lst = []
    scenario_counter = 1
    scenario_ids = []
    for scenario_id in scenario_dict:
        scenario = scenario_dict[scenario_id]
        scenario_data = [
            scenario_counter,
            scenario_id,
            len(scenario.timestamps_seconds),
            len(scenario.tracks),
            len(scenario.dynamic_map_states),
            len(scenario.objects_of_interest),
        ]
        scenario_ids.append(scenario_id)
        scenario_data_lst.append(scenario_data)
        scenario_counter += 1
    print("                                               ")
    print("-----------------------------------------------")
    print(
        f"{len(scenario_dict)} scenarios in {tfrecord_path}:\n"
    )
    print(
        tabulate(
            scenario_data_lst,
            headers=[
                "Index",
                "Scenario ID",
                "Timestamps",
                "Track Objects",
                "Traffic Lights",
                "Object of Interest",
            ],
        )
    )
    return scenario_ids


def export_scenario(
    target_base_path: str, tfrecord_file_path: str, scenario_id
):
    subfolder_path = os.path.join(target_base_path, scenario_id)
    try:
        os.makedirs(subfolder_path)
    except FileExistsError:
        print(f"Folder already exists at path {subfolder_path}")
    scenario_py = os.path.join(subfolder_path, "scenario.py")
    if os.path.exists(scenario_py):
        print(f"scenario.py already exists in {subfolder_path}.")
    else:
        scenario_template = os.path.join(
            Path(__file__).parent, "templates", "scenario_template.py"
        )
        shutil.copy2(scenario_template, scenario_py)
        print(f"Scenario.py created in {subfolder_path}.")

    yaml_dataspec = {
        "trajectory_dataset": {
            "source": "Waymo",
            "input_path": tfrecord_file_path,
            "scenario_id": scenario_id,
        }
    }
    with open(os.path.join(subfolder_path, "waymo.yaml"), "w") as yaml_file:
        yaml.dump(yaml_dataspec, yaml_file, default_flow_style=False)


def tfrecords_browser(tfrecord_path: str):
    scenarios_per_tfrecords = parse_tfrecords(tfrecord_path)
    tf_records = []
    tf_counter = 1
    for tf in scenarios_per_tfrecords:
        tf_records.append([tf_counter, tf])
        tf_counter += 1
    stop_browser = False
    while not stop_browser:
        print("-----------------------------------------------")
        print("Waymo tfRecords:\n")
        print(tabulate(tf_records, headers=["Index", "TfRecords"],))
        print("\n")
        print("You can use the following commands to further explore these datasets:\n"
              "1. `display all` --> Displays the info of all the scenarios from every tfRecord file together\n"
              f"2. `explore <index>` --> Explore the tfRecord file at this index of the table. The index should be an integer between 1 and {tf_counter}\n"
              "3. `exit` --> Exit the program\n")

        raw_input = input("Command: ").lower()
        user_input = raw_input.strip()
        if user_input == "display all":
            for tf_record in tf_records:
                display_scenarios_in_tfrecord(tf_record, scenarios_per_tfrecords[tf_record])
        elif user_input == "exit":
            stop_browser = True
            print("Exiting Browser")
        elif "explore" in user_input:
            input_lst = user_input.split()
            if len(input_lst) != 2:
                print("Please enter only one number as an index for the `explore` command")
                continue
            try:
                idx = int(input_lst[1])
                if not (1 <= idx <= len(tf_records)):
                    print(f"Please enter an index between 1 and {tf_counter}.")
                    continue
                tf_path = tf_records[idx][1]
                stop_browser = explore_tf_record(tf_path, scenarios_per_tfrecords[tf_path])
            except Exception:
                print("Please input an integer for the `explore` command")
                continue
        else:
            print("Please enter a valid command. See command formats above")


def explore_tf_record(tfrecord: str, scenario_dict):
    scenario_ids = display_scenarios_in_tfrecord(tfrecord, scenario_dict)
    print("\n")
    print("You can use the following commands to further explore these scenarios:\n"
          "1. `export all <target_base_path>` --> Export all scenarios in this tf_record to a target path. Path should be valid.\n"
          f"2. `export <index> <target_base_path>' --> Export the scenario at this index of the table to a target path. The index should be an integer between 1 and {len(scenario_ids)} and path should be valid.\n"
          "3. `preview all <target_base_path>` --> Plot and dump the images of the map of all scenarios in this tf_record to a target path. Path should be valid.\n"
          f"4. `preview <index>` --> Plot and display the map of the scenario at this index of the table. The index should be an integer between 1 and {len(scenario_ids)}\n"
          f"5. `select <index>` --> Select and explore further the scenario at this index of the table. The index should be an integer between 1 and {len(scenario_ids)}\n"
          "6. `go back` --> Go back to the tfrecords browser\n"
          "7. `exit` --> Exit the program\n")

    raw_input = input("Command: ").lower()
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="waymo_scenario_gen.py",
        description="Extract map data from Waymo dataset, plot the scenarios and save their ids.",
    )
    parser.add_argument("file", help="TFRecord file/folder path")
    args = parser.parse_args()

    # display_scenario_info(parse_tfrecords(args.file))
    export_scenario("scenarios/waymo_motion", args.file, "4f30f060069bbeb9")
