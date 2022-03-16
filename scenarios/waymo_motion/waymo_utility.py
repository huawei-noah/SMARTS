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
import re
from typing import Dict, List, Tuple, Union
from tabulate import tabulate
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from waymo_open_dataset.protos import scenario_pb2

from smarts.core.utils.file import read_tfrecord_file


def convert_polyline(polyline) -> Tuple[List[float], List[float]]:
    xs, ys = [], []
    for p in polyline:
        xs.append(p.x)
        ys.append(p.y)
    return xs, ys


def get_map_features_for_scenario(scenario) -> Dict:
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
            map_features[key].append(
                getattr(map_feature, key)
            )  # tls_lanes = get_traffic_light_lanes(scenario)
    return map_features


def get_traffic_light_lanes(scenario) -> List[str]:
    num_steps = len(scenario.timestamps_seconds)
    tls_lanes = []
    for i in range(num_steps):
        dynamic_states = scenario.dynamic_map_states[i]
        for j in range(len(dynamic_states.lane_states)):
            lane_state = dynamic_states.lane_states[j]
            tls_lanes.append(lane_state.lane)
    return tls_lanes


def plot_map_features(map_features) -> List[Line2D]:
    handles = []
    lanes = map_features["lane"]
    lane_points = [convert_polyline(lane.polyline) for lane in lanes]
    # lanes = list(filter(lambda lane: max(lane[1]) > 8150, lanes))
    for xs, ys in lane_points:
        plt.plot(xs, ys, linestyle=":", color="gray")
    handles.append(Line2D([0], [0], linestyle=":", color="gray", label="Lane Polyline"))

    for road_line in map_features["road_line"]:
        xs, ys = convert_polyline(road_line.polyline)
        if road_line.type in [1, 4, 5]:
            plt.plot(xs, ys, "y--")
        else:
            plt.plot(xs, ys, "y-")
    handles.append(
        Line2D([0], [0], linestyle="-", color="yellow", label="Single Road Line")
    )
    handles.append(
        Line2D([0], [0], linestyle="--", color="yellow", label="Double Road Line")
    )

    for road_edge in map_features["road_edge"]:
        xs, ys = convert_polyline(road_edge.polyline)
        plt.plot(xs, ys, "k-")
    handles.append(Line2D([0], [0], linestyle="-", color="black", label="Road Edge"))

    for crosswalk in map_features["crosswalk"]:
        xs, ys = convert_polyline(crosswalk.polygon)
        plt.plot(xs, ys, "k--")
    handles.append(Line2D([0], [0], linestyle="--", color="black", label="Crosswalk"))

    for speed_bump in map_features["speed_bump"]:
        xs, ys = convert_polyline(speed_bump.polygon)
        plt.plot(xs, ys, "k:")
    handles.append(Line2D([0], [0], linestyle=":", color="black", label="Speed Bump"))

    for stop_sign in map_features["stop_sign"]:
        plt.scatter(
            stop_sign.position.x, stop_sign.position.y, marker="o", c="#ff0000", alpha=1
        )
    handles.append(
        Line2D(
            [],
            [],
            color="red",
            marker="o",
            linestyle="None",
            markersize=5,
            label="Stop Sign",
        )
    )
    return handles


def plot_scenario(scenario):
    # Get map feature data from map proto
    map_features = get_map_features_for_scenario(scenario)

    # Plot map
    fig, ax = plt.subplots()
    ax.set_title(f"Scenario {scenario.scenario_id}")
    ax.axis("equal")
    handles = plot_map_features(map_features)
    plt.legend(handles=handles)

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
        handles = plot_map_features(map_features)
        plt.legend(handles=handles)
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
    print(f"{len(scenario_dict)} scenarios in {tfrecord_path}:\n")
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
) -> None:
    subfolder_path = os.path.join(target_base_path, scenario_id)
    try:
        os.makedirs(subfolder_path)
        print(f"Created folder {scenario_id} at path {target_base_path}")
    except FileExistsError:
        print(f"Folder already exists at path {subfolder_path}")
    except (OSError, RuntimeError):
        print(f"{target_base_path} is an invalid path. Please enter a valid path")
        return
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
        print(f"waymo.yaml created in {subfolder_path}")


def tfrecords_browser(tfrecord_path: str):
    scenarios_per_tfrecords = parse_tfrecords(tfrecord_path)
    tf_records = []
    tf_counter = 1
    for tf in scenarios_per_tfrecords:
        tf_records.append([tf_counter, tf])
        tf_counter += 1
    stop_browser = False
    while not stop_browser:
        print("\n")
        print("-----------------------------------------------")
        print("Waymo tfRecords:\n")
        print(
            tabulate(
                tf_records,
                headers=["Index", "TfRecords"],
            )
        )
        print("\n")
        print(
            "You can use the following commands to further explore these datasets:\n"
            "1. `display all` --> Displays the info of all the scenarios from every tfRecord file together\n"
            f"2. `explore <index>` --> Explore the tfRecord file at this index of the table. The index should be an integer between 1 and {len(tf_records)}\n"
            "3. `exit` --> Exit the program\n"
        )
        print("\n")
        raw_input = input("Command: ").lower()
        user_input = raw_input.strip()
        if user_input == "display all":
            for tf_record in tf_records:
                display_scenarios_in_tfrecord(
                    tf_record[1], scenarios_per_tfrecords[tf_record[1]]
                )
        elif user_input == "exit":
            stop_browser = True
            print("Exiting the Browser")

        elif re.compile("^explore [\d]+$").match(user_input):
            input_lst = user_input.split()
            try:
                idx = int(input_lst[1])
                if not (1 <= idx <= len(tf_records)):
                    print(f"Please enter an index between 1 and {len(tf_records)}.")
                    continue
            except Exception:
                print("Please input an integer for the the `explore` command")
                continue
            tf_path = tf_records[idx][1]
            stop_browser = explore_tf_record(
                tf_path, scenarios_per_tfrecords[tf_path]
            )
        else:
            print("Please enter a valid command. See command formats above")


def explore_tf_record(tfrecord: str, scenario_dict):
    scenario_ids = display_scenarios_in_tfrecord(tfrecord, scenario_dict)
    stop_exploring = False
    while not stop_exploring:
        print("\n")
        print(
            "You can use the following commands to further explore these scenarios:\n"
            "1. `export all <target_base_path>` --> Export all scenarios in this tf_record to a target path. Path should be valid directory path.\n"
            f"2. `export <index> <target_base_path>' --> Export the scenario at this index of the table to a target path. The index should be an integer between 1 and {len(scenario_ids)} and path should be valid.\n"
            "3. `preview all <target_base_path>` --> Plot and dump the images of the map of all scenarios in this tf_record to a target path. Path should be valid.\n"
            f"4. `preview <index>` --> Plot and display the map of the scenario at this index of the table. The index should be an integer between 1 and {len(scenario_ids)}\n"
            f"5. `select <index>` --> Select and explore further the scenario at this index of the table. The index should be an integer between 1 and {len(scenario_ids)}\n"
            "6. `go back` --> Go back to the tfrecords browser\n"
            "7. `exit` --> Exit the program\n"
        )
        print("\n")
        raw_input = input("Command: ").lower()
        user_input = raw_input.strip()
        if re.compile("^export all [^\n ]+$").match(user_input):
            target_base_path = user_input.split()[2]
            # Check if target base path is valid
            try:
                Path(target_base_path).resolve()
            except (OSError, RuntimeError):
                print(
                    f"{target_base_path} is an invalid path. Please enter a valid directory path"
                )
                continue
            # Try exporting all the scenarios
            for id in scenario_ids:
                export_scenario(target_base_path, tfrecord, id)

        elif re.compile("^export [\d]+ [^\n ]+$").match(user_input):
            input_lst = user_input.split()

            # Check if index passed is valid
            try:
                scenario_idx = int(input_lst[1])
                if not (1 <= scenario_idx <= len(scenario_ids)):
                    print(f"Please enter an index between 1 and {len(scenario_ids)}.")
                    continue
            except Exception:
                print("Please input an integer for the index argument of `export` command")
                continue
            # Check if target base path is valid
            target_base_path = input_lst[2]
            try:
                Path(target_base_path).resolve()
            except (OSError, RuntimeError):
                print(
                    f"{target_base_path} is an invalid path. Please enter a valid directory path"
                )
                continue
            # Try exporting the scenario
            export_scenario(target_base_path, tfrecord, scenario_ids[scenario_idx])

        elif re.compile("^preview all [^\n ]+$").match(user_input):
            input_lst = user_input.split()

            # Check if target base path is valid
            target_base_path = input_lst[2]
            try:
                Path(target_base_path).resolve()
            except (OSError, RuntimeError):
                print(
                    f"{target_base_path} is an invalid path. Please enter a valid path"
                )
                continue
            # Dump all the scenario plots of this tfrecord file to this target base path
            dump_plots(target_base_path, tfrecord)

        elif re.compile("^preview [\d]+$").match(user_input):
            input_lst = user_input.split()

            # Check if index passed is valid
            try:
                scenario_idx = int(input_lst[1])
                if not (1 <= scenario_idx <= len(scenario_ids)):
                    print(f"Please enter an index between 1 and {len(scenario_ids)}.")
                    continue
            except Exception:
                print("Please input an integer for the index argument of `export` command")
                continue
            # Dump all the scenario plots of this tfrecord file to this target base path
            scenario_id = scenario_ids[scenario_idx]
            plot_scenario(scenario_dict[scenario_id])

        elif re.compile("^select [\d]+$").match(user_input):
            input_lst = user_input.split()

            # Check if index passed is valid
            scenario_idx = int(input_lst[1])
            if not (1 <= scenario_idx <= len(scenario_ids)):
                print(f"Please enter an index between 1 and {len(scenario_ids)}.")
                continue

            # Explore further the scenario at this index
            scenario_id = scenario_ids[scenario_idx]
            # stop_exploring = explore_scenario(scenario_dict[scenario_id])

        elif user_input == "go back":
            stop_exploring = True
            print("Going back to the tfRecords browser")
            continue

        elif user_input == "exit":
            return True
        else:
            print("Please enter a valid command. See command formats above")
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="waymo_scenario_gen.py",
        description="Extract map data from Waymo dataset, plot the scenarios and save their ids.",
    )
    parser.add_argument("file", help="TFRecord file/folder path")
    args = parser.parse_args()

    # display_scenario_info(parse_tfrecords(args.file))
    # export_scenario("scenarios/waymo_motion", args.file, "4f30f060069bbeb9")
    tfrecords_browser(args.file)
