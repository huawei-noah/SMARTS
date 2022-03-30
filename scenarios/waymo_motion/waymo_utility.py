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

# Text based Waymo Dataset Browser.
import argparse
import copy
import os
import shutil
import yaml
import re
import json
from typing import Dict, List, Tuple, Optional
from tabulate import tabulate
from pathlib import Path
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from waymo_open_dataset.protos import scenario_pb2

from smarts.core.utils.file import read_tfrecord_file


def lerp(a: float, b: float, t: float) -> float:
    return t * (b - a) + a


def convert_polyline(polyline) -> Tuple[List[float], List[float]]:
    xs, ys = [], []
    for p in polyline:
        xs.append(p.x)
        ys.append(p.y)
    return xs, ys


def get_trajectory_handles() -> List[Line2D]:
    handles = [
        Line2D(
            [],
            [],
            color="cyan",
            marker="^",
            linestyle="None",
            markersize=5,
            label="Ego Vehicle",
        ),
        Line2D(
            [],
            [],
            color="black",
            marker="^",
            linestyle="None",
            markersize=5,
            label="Car",
        ),
        Line2D(
            [],
            [],
            color="magenta",
            marker="d",
            linestyle="None",
            markersize=5,
            label="Pedestrian",
        ),
        Line2D(
            [],
            [],
            color="green",
            marker="*",
            linestyle="None",
            markersize=5,
            label="Cyclist",
        ),
        Line2D(
            [],
            [],
            color="black",
            marker="8",
            linestyle="None",
            markersize=5,
            label="Other",
        ),
    ]
    return handles


def get_map_handles() -> List[Line2D]:
    handles = [
        Line2D([0], [0], linestyle=":", color="gray", label="Lane Polyline"),
        Line2D([0], [0], linestyle="-", color="yellow", label="Single Road Line"),
        Line2D([0], [0], linestyle="--", color="yellow", label="Double Road Line"),
        Line2D([0], [0], linestyle="-", color="black", label="Road Edge"),
        Line2D([0], [0], linestyle="--", color="black", label="Crosswalk"),
        Line2D([0], [0], linestyle=":", color="black", label="Speed Bump"),
        Line2D(
            [],
            [],
            color="red",
            marker="o",
            linestyle="None",
            markersize=5,
            label="Stop Sign",
        ),
    ]
    return handles


def get_map_features_for_scenario(scenario: scenario_pb2.Scenario) -> Dict:
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
            map_features[key].append((getattr(map_feature, key), str(map_feature.id)))

    return map_features


def get_traffic_light_lanes(scenario: scenario_pb2.Scenario) -> List[str]:
    num_steps = len(scenario.timestamps_seconds)
    tls_lanes = []
    for i in range(num_steps):
        dynamic_states = scenario.dynamic_map_states[i]
        for j in range(len(dynamic_states.lane_states)):
            lane_state = dynamic_states.lane_states[j]
            tls_lanes.append(lane_state.lane)
    return tls_lanes


def get_object_type_count(trajectories):
    cars, pedestrian, cyclist, other = [], [], [], []
    ego = None
    for vehicle_id in trajectories:
        if trajectories[vehicle_id][2] == 1:
            ego = vehicle_id
        elif trajectories[vehicle_id][3] == 1:
            cars.append(vehicle_id)
        elif trajectories[vehicle_id][3] == 2:
            pedestrian.append(vehicle_id)
        elif trajectories[vehicle_id][3] == 3:
            cyclist.append(vehicle_id)
        else:
            other.append(vehicle_id)
    return ego, cars, pedestrian, cyclist, other


def get_trajectory_data(waymo_scenario: scenario_pb2.Scenario):
    def generate_trajectory_rows(scenario):
        for i in range(len(scenario.tracks)):
            vehicle_id = scenario.tracks[i].id
            num_steps = len(scenario.timestamps_seconds)
            rows = []

            # First pass -- extract data
            for j in range(num_steps):
                obj_state = scenario.tracks[i].states[j]
                row = dict()
                row["vehicle_id"] = vehicle_id
                row["type"] = scenario.tracks[i].object_type
                row["is_ego_vehicle"] = 1 if i == scenario.sdc_track_index else 0
                row["valid"] = obj_state.valid
                row["sim_time"] = scenario.timestamps_seconds[j]
                row["position_x"] = obj_state.center_x
                row["position_y"] = obj_state.center_y
                rows.append(row)

            # Second pass -- align timesteps to 10 Hz and interpolate trajectory data if needed
            interp_rows = [None] * num_steps
            for j in range(num_steps):
                row = rows[j]
                timestep = 0.1
                time_current = row["sim_time"]
                time_expected = round(j * timestep, 3)
                time_error = time_current - time_expected

                if not row["valid"] or time_error == 0:
                    continue

                if time_error > 0:
                    # We can't interpolate if the previous element doesn't exist or is invalid
                    if j == 0 or not rows[j - 1]["valid"]:
                        continue

                    # Interpolate backwards using previous timestep
                    interp_row = {"sim_time": time_expected}

                    prev_row = rows[j - 1]
                    prev_time = prev_row["sim_time"]

                    t = (time_expected - prev_time) / (time_current - prev_time)
                    interp_row["position_x"] = lerp(
                        prev_row["position_x"], row["position_x"], t
                    )
                    interp_row["position_y"] = lerp(
                        prev_row["position_y"], row["position_y"], t
                    )
                    interp_rows[j] = interp_row
                else:
                    # We can't interpolate if the next element doesn't exist or is invalid
                    if (
                        j == len(scenario.timestamps_seconds) - 1
                        or not rows[j + 1]["valid"]
                    ):
                        continue

                    # Interpolate forwards using next timestep
                    interp_row = {"sim_time": time_expected}

                    next_row = rows[j + 1]
                    next_time = next_row["sim_time"]

                    t = (time_expected - time_current) / (next_time - time_current)
                    interp_row["position_x"] = lerp(
                        row["position_x"], next_row["position_x"], t
                    )
                    interp_row["position_y"] = lerp(
                        row["position_y"], next_row["position_y"], t
                    )
                    interp_rows[j] = interp_row

            # Third pass -- filter invalid states, replace interpolated values
            for j in range(num_steps):
                if not rows[j]["valid"]:
                    continue
                if interp_rows[j] is not None:
                    rows[j]["sim_time"] = interp_rows[j]["sim_time"]
                    rows[j]["position_x"] = interp_rows[j]["position_x"]
                    rows[j]["position_y"] = interp_rows[j]["position_y"]
                yield rows[j]

    trajectories = {}
    agent_id = None
    for t_row in generate_trajectory_rows(waymo_scenario):
        if agent_id != t_row["vehicle_id"]:
            agent_id = t_row["vehicle_id"]
            trajectories[agent_id] = [
                [],
                [],
                t_row["is_ego_vehicle"],
                t_row["type"],
            ]
        trajectories[agent_id][0].append(t_row["position_x"])
        trajectories[agent_id][1].append(t_row["position_y"])
    return trajectories


def plot_map_features(map_features, feature_ids: List[str]) -> List[Line2D]:
    handles = []
    for lane in map_features["lane"]:
        xs, ys = convert_polyline(lane[0].polyline)
        if lane[1] in feature_ids:
            plt.plot(xs, ys, linestyle=":", color="blue", linewidth=2.0)
            handles.append(
                Line2D(
                    [0],
                    [0],
                    linestyle=":",
                    color="blue",
                    label=f"Lane Polyline {lane[1]}",
                ),
            )
        else:
            plt.plot(xs, ys, linestyle=":", color="gray")

    for road_line in map_features["road_line"]:
        xs, ys = convert_polyline(road_line[0].polyline)
        if road_line[0].type in [1, 4, 5]:
            if road_line[1] in feature_ids:
                plt.plot(xs, ys, "b--", linewidth=2.0)
                handles.append(
                    Line2D(
                        [0],
                        [0],
                        linestyle="--",
                        color="blue",
                        label=f"Single Road Line {road_line[1]}",
                    )
                )
            else:
                plt.plot(xs, ys, "y--")
        else:
            if road_line[1] in feature_ids:
                plt.plot(xs, ys, "b-", linewidth=2.0)
                handles.append(
                    Line2D(
                        [0],
                        [0],
                        linestyle="-",
                        color="blue",
                        label=f"Double Road Line {road_line[1]}",
                    )
                )
            else:
                plt.plot(xs, ys, "y-")

    for road_edge in map_features["road_edge"]:
        xs, ys = convert_polyline(road_edge[0].polyline)
        if road_edge[1] in feature_ids:
            plt.plot(xs, ys, "b-", linewidth=2.0)
            handles.append(
                Line2D(
                    [0],
                    [0],
                    linestyle="-",
                    color="blue",
                    label=f"Road Edge {road_edge[1]}",
                )
            )
        else:
            plt.plot(xs, ys, "k-")

    for crosswalk in map_features["crosswalk"]:
        xs, ys = convert_polyline(crosswalk[0].polygon)
        if crosswalk[1] in feature_ids:
            plt.plot(xs, ys, "b--", linewidth=2.0)
            handles.append(
                Line2D(
                    [0],
                    [0],
                    linestyle="--",
                    color="blue",
                    label=f"Crosswalk {crosswalk[1]}",
                )
            )
        else:
            plt.plot(xs, ys, "k--")

    for speed_bump in map_features["speed_bump"]:
        xs, ys = convert_polyline(speed_bump[0].polygon)
        if speed_bump[1] in feature_ids:
            plt.plot(xs, ys, "b:", linewidth=2.0)
            handles.append(
                Line2D(
                    [0],
                    [0],
                    linestyle=":",
                    color="blue",
                    label=f"Speed Bump {speed_bump[1]}",
                )
            )
        else:
            plt.plot(xs, ys, "k:")

    for stop_sign in map_features["stop_sign"]:
        if stop_sign[1] in feature_ids:
            s_color = "blue"
            handles.append(
                Line2D(
                    [],
                    [],
                    color="blue",
                    marker="o",
                    markersize=5,
                    label=f"Stop Sign {stop_sign[1]}",
                )
            )
        else:
            s_color = "#ff0000"
        plt.scatter(
            stop_sign[0].position.x,
            stop_sign[0].position.y,
            marker="o",
            c=s_color,
            alpha=1,
        )
    return handles


def plot_trajectories(trajectories, track_ids: List[str]):
    handles = []
    max_len = 0
    data, points = [], []
    for k, v in trajectories.items():
        xs, ys = v[0], v[1]
        is_ego = v[2]
        object_type = v[3]
        if len(xs) > max_len:
            max_len = len(xs)
        if is_ego:
            (point,) = plt.plot(xs[0], ys[0], "c^")
        elif object_type == 1:
            if str(k) in track_ids:
                (point,) = plt.plot(xs[0], ys[0], "r^")
                handles.append(
                    Line2D(
                        [],
                        [],
                        color="red",
                        marker="^",
                        linestyle="None",
                        markersize=5,
                        label=f"Car {k}",
                    )
                )
            else:
                (point,) = plt.plot(xs[0], ys[0], "k^")
        elif object_type == 2:
            if str(k) in track_ids:
                (point,) = plt.plot(xs[0], ys[0], "rd")
                handles.append(
                    Line2D(
                        [],
                        [],
                        color="red",
                        marker="d",
                        linestyle="None",
                        markersize=5,
                        label=f"Pedestrian {k}",
                    )
                )
            else:
                (point,) = plt.plot(xs[0], ys[0], "md")
        elif object_type == 3:
            if str(k) in track_ids:
                (point,) = plt.plot(xs[0], ys[0], "r*")
                handles.append(
                    Line2D(
                        [],
                        [],
                        color="red",
                        marker="*",
                        linestyle="None",
                        markersize=5,
                        label=f"Cyclist {k}",
                    )
                )
            else:
                (point,) = plt.plot(xs[0], ys[0], "g*")
        else:
            if str(k) in track_ids:
                (point,) = plt.plot(xs[0], ys[0], "r8")
                handles.append(
                    Line2D(
                        [],
                        [],
                        color="red",
                        marker="8",
                        linestyle="None",
                        markersize=5,
                        label=f"Other {k}",
                    )
                )
            else:
                (point,) = plt.plot(xs[0], ys[0], "k8")
        data.append((xs, ys))
        points.append(point)

    return data, points, max_len, handles


def plot_scenarios(
    scenario_infos, animate: bool, feature_ids: Optional[List[str]] = None
):
    def plot_map_and_get_animate(scenario_info, animate_trajectories, fig_num, f_ids):
        anim = None
        # Get map feature data from map proto
        map_features = scenario_info[1]

        # Plot map
        fig = plt.figure(num=fig_num)
        if animate_trajectories or not f_ids:
            highlighted_handles = plot_map_features(map_features, [])
        else:
            highlighted_handles = plot_map_features(map_features, f_ids)
        plt.title(f"Scenario {scenario_info[0].scenario_id}")

        # Set Legend Handles
        all_handles = get_map_handles()
        all_handles.extend(highlighted_handles)

        # Resize figure
        mng = plt.get_current_fig_manager()
        mng.resize(1000, 1000)

        if animate_trajectories:
            # Plot Trajectories
            data, points, max_len, t_handles = plot_trajectories(
                scenario_info[2], f_ids if f_ids else []
            )
            all_handles.extend(get_trajectory_handles() + t_handles)

            def update(i):
                drawn_pts = []
                for (xs, ys), point in zip(data, points):
                    if i < len(xs):
                        point.set_data(xs[i], ys[i])
                        drawn_pts.append(point)
                return drawn_pts

            # Set Animation
            anim = FuncAnimation(fig, update, frames=max_len, blit=True, interval=100)
        plt.legend(handles=all_handles)
        return anim

    animates = []
    counter = 1
    for s_info in scenario_infos:
        animates.append(plot_map_and_get_animate(s_info, animate, counter, feature_ids))
        counter += 1
    plt.show()


def dump_plots(target_base_path: str, scenario_dict, animate=False):
    try:
        os.makedirs(os.path.abspath(target_base_path))
        print(f"Created directory {target_base_path}")
    except FileExistsError:
        pass
    except (OSError, RuntimeError):
        print(f"{target_base_path} is an invalid path. Please enter a valid path")
        return
    for scenario_id in scenario_dict:
        scenario = scenario_dict[scenario_id][0]

        fig = plt.figure()
        mng = plt.get_current_fig_manager()
        plt.title(f"Scenario {scenario_id}")

        # Resize figure
        w = 1000
        h = 1000
        mng.resize(w, h)

        # Plot map
        if scenario_dict[scenario_id][1] is None:
            scenario_dict[scenario_id][1] = get_map_features_for_scenario(scenario)
        plot_map_features(scenario_dict[scenario_id][1], [])
        all_handles = get_map_handles()

        if animate:
            # Plot Trajectories
            if scenario_dict[scenario_id][2] is None:
                scenario_dict[scenario_id][2] = get_map_features_for_scenario(scenario)
            data, points, max_len, _ = plot_trajectories(
                scenario_dict[scenario_id][2], []
            )
            all_handles.extend(get_trajectory_handles())
            plt.legend(handles=all_handles)

            def update(i):
                drawn_pts = []
                for (xs, ys), point in zip(data, points):
                    if i < len(xs):
                        point.set_data(xs[i], ys[i])
                        drawn_pts.append(point)
                return drawn_pts

            # Set Animation
            anim = FuncAnimation(fig, update, frames=max_len, blit=True, interval=100)
            out_path = os.path.join(
                os.path.abspath(target_base_path), f"scenario-{scenario_id}.mp4"
            )
            anim.save(out_path, writer=FFMpegWriter(fps=15))

        else:
            plt.legend(handles=all_handles)
            out_path = os.path.join(
                os.path.abspath(target_base_path), f"scenario-{scenario_id}.png"
            )
            fig = plt.gcf()
            fig.set_size_inches(w / 100, h / 100)
            fig.savefig(out_path, dpi=100)

        print(f"Saving {out_path}")
        plt.close("all")
    print(f"All images or recordings saved at {target_base_path}")


def display_scenario_tags(tags_per_scenarios: Dict[str, List[str]]):
    tag_data = []
    for scenario_id in tags_per_scenarios:
        tag_data.append([scenario_id, tags_per_scenarios[scenario_id]])
    print(
        tabulate(
            tag_data,
            headers=[
                "Scenario ID",
                "Tags",
            ],
        )
    )
    print("\n")


def merge_tags(new_imports, main_dict, display: bool = False):
    for tf_file in new_imports:
        if tf_file in main_dict:
            for scenario_id in new_imports[tf_file]:
                if scenario_id in main_dict:
                    main_dict[tf_file][scenario_id].extend(
                        tag.lower()
                        for tag in new_imports[tf_file][scenario_id]
                        if tag.lower() not in main_dict[tf_file][scenario_id]
                    )
                else:
                    main_dict[tf_file][scenario_id] = new_imports[tf_file][scenario_id]
        else:
            main_dict[tf_file] = new_imports[tf_file]
        if display:
            print("\n-----------------------------------------------")
            print(f"Scenario Tags imported for {tf_file}:\n")
            display_scenario_tags(new_imports[tf_file])


def get_scenario_and_tag_dict(tfrecord_file: str):
    scenario_dict = {}
    tags_per_scenario = {}
    dataset = read_tfrecord_file(tfrecord_file)
    for record in dataset:
        scenario = scenario_pb2.Scenario()
        scenario.ParseFromString(bytearray(record))
        scenario_id = scenario.scenario_id
        scenario_dict[scenario_id] = [scenario, None, None]
        tags_per_scenario[scenario_id] = []
    return scenario_dict, tags_per_scenario


def parse_tfrecords(tfrecord_paths: List[str]):
    scenarios_per_tfrecord, tags_per_tfrecord = {}, {}
    for tfrecord_path in tfrecord_paths:
        if os.path.isdir(tfrecord_path):
            for root, dirs, files in os.walk(tfrecord_path):
                for file in files:
                    if ".tfrecord" in file:
                        scenarios_per_tfrecord[os.path.join(root, file)] = None
                        tags_per_tfrecord[file] = None
        elif ".tfrecord" in tfrecord_path:
            scenarios_per_tfrecord[tfrecord_path] = None
            tags_per_tfrecord[os.path.basename(tfrecord_path)] = None
    return scenarios_per_tfrecord, tags_per_tfrecord


def display_tf_records(records):
    print("\n-----------------------------------------------")
    print("Waymo tfRecords:\n")
    print(
        tabulate(
            records,
            headers=["Index", "TfRecords"],
        )
    )
    print("\n\n")


def display_scenarios_in_tfrecord(
    tfrecord_path: str, scenario_dict, tfrecord_tags, tags_imported
) -> List[str]:
    scenario_data_lst = []
    scenario_counter = 1
    scenario_ids = []
    for scenario_id in scenario_dict:
        scenario = scenario_dict[scenario_id][0]
        scenario_data = [
            scenario_counter,
            scenario_id,
            len(scenario.timestamps_seconds),
            len(scenario.tracks),
            len(scenario.dynamic_map_states),
            len(scenario.objects_of_interest),
            tfrecord_tags.get(scenario_id, []),
            tags_imported.get(scenario_id, []),
        ]
        scenario_ids.append(scenario_id)
        scenario_data_lst.append(scenario_data)
        scenario_counter += 1
    print("\n-----------------------------------------------")
    print(f"{len(scenario_dict)} scenarios in {tfrecord_path}:\n")
    print(
        tabulate(
            scenario_data_lst,
            headers=[
                "Index",
                "Scenario ID",
                "Timestamps",
                "Track Objects",
                "Traffic Light States",
                "Objects of Interest",
                "Tags Added",
                "Tags Imported",
            ],
        )
    )
    return scenario_ids


def export_scenario(
    target_base_path: str, tfrecord_file_path: str, scenario_id: str
) -> bool:
    subfolder_path = os.path.join(os.path.abspath(target_base_path), scenario_id)
    try:
        os.makedirs(subfolder_path)
        print(f"Created folder {scenario_id} at path {target_base_path}")
    except FileExistsError:
        print(f"Folder {scenario_id} already exists at path {target_base_path}")
    except (OSError, RuntimeError):
        print(f"{target_base_path} is an invalid path. Please enter a valid path")
        return False
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
    print("\n")
    return True


def check_index_validity(
    input_arg: List[str], upper_limit: int, command_type: str
) -> List[int]:
    valid_indexes = []
    for input_index in input_arg:
        try:
            idx = int(input_index)
            if not (1 <= idx <= upper_limit):
                print(
                    f"{input_index} is out of bound. Please enter an index between 1 and {upper_limit}."
                )
            valid_indexes.append(idx)

        except ValueError:
            print(
                f"{valid_indexes} is Invalid index. Please input integers as index for the `{command_type}` command"
            )
    if not valid_indexes:
        print(
            "no valid indexes passed. Please check the command info for valid index values"
        )
    return list(set(valid_indexes))


def check_path_validity(target_base_path: str) -> bool:
    # Check if target base path is valid
    try:
        Path(target_base_path).resolve()
    except (IOError, OSError, RuntimeError):
        print(
            f"{target_base_path} is an invalid path. Please enter a valid directory path"
        )
        return False
    return True


def tfrecords_browser(
    tfrecord_paths: List[str], default_target_path: Optional[str] = None
) -> None:
    scenarios_per_tfrecords, tags_per_tfrecords = parse_tfrecords(tfrecord_paths)
    imported_tags = {}
    if not scenarios_per_tfrecords:
        print("No .tfrecord files exist in paths provided. Please pass valid paths.")
        return

    tf_records = []
    tf_counter = 1
    for tf in scenarios_per_tfrecords:
        tf_records.append([tf_counter, tf])
        tf_counter += 1
    stop_browser = False

    display_tf_records(tf_records)
    print_commands = True
    while not stop_browser:
        if print_commands:
            print(
                "TfRecords Browser.\n"
                "You can use the following commands to further explore these datasets:\n"
                "1. `display all` --> Displays the info of all the scenarios from every tfRecord file together\n"
                f"2. `display <indexes>` --> Displays the info of tfRecord files at these indexes of the table.\n"
                f"                           The indexes should be an integer between 1 and {len(tf_records)} and space separated\n"
                f"3. `import tags` --> Import the tags of tfRecords from a previously saved .json file.\n"
                f"                                      The path to the .json file should be valid and exist.\n"
                f"                                      Only tags of tfRecords which are displayed above will be imported. Ensure the name of tfRecord match with the ones displayed above.\n"
                f"4. `export tags all/<indexes>` --> Export the tags of the tfRecords at these indexes to a .json file.\n"
                f"                                                    Optionally if you can use all instead to export tags of all tfRecords. The path to the .json file should be valid.\n"
                f"5. `explore <index>` --> Explore the tfRecord file at this index of the table.\n"
                f"                         The index should be an integer between 1 and {len(tf_records)}\n"
                "6. `exit` --> Exit the program\n"
            )

            print_commands = False
        print("\n")
        try:
            raw_input = input("\nCommand: ")
            user_input = raw_input.strip()
        except EOFError:
            print("Raised EOF. Attempting to exit browser.")
            stop_browser = True
            continue

        if re.compile("^display[\s]+(all|(?:\s*(\d+))+)$", flags=re.IGNORECASE).match(
            user_input
        ):
            input_lst = user_input.split()
            if input_lst[1].lower() == "all":
                valid_indexes = [i + 1 for i in range(len(tf_records))]
            else:
                valid_indexes = check_index_validity(
                    input_lst[1:], len(tf_records), "display"
                )
                if len(valid_indexes) == 0:
                    continue
            for idx in valid_indexes:
                tfr_path = tf_records[idx - 1][1]
                if scenarios_per_tfrecords[tfr_path] is None:
                    (
                        scenarios_per_tfrecords[tfr_path],
                        tags_per_tfrecords[os.path.basename(tfr_path)],
                    ) = get_scenario_and_tag_dict(tfr_path)
                display_scenarios_in_tfrecord(
                    tfr_path,
                    scenarios_per_tfrecords[tfr_path],
                    tags_per_tfrecords[os.path.basename(tfr_path)],
                    imported_tags.get(os.path.basename(tfr_path), {}),
                )
                print("\n")
            print_commands = True

        elif re.compile("^import[\s]+tags$", flags=re.IGNORECASE).match(user_input):
            valid_path = False
            print(
                "Enter the path to .json file to which you want to import the tags from?:\n"
            )
            new_tags = {}
            stripped_path = None
            while not valid_path:
                try:
                    response = input("\nEnter Path: ")
                    stripped_path = response.strip("[ \"']")
                except EOFError:
                    print("Raised EOF. Attempting to exit browser.")
                    stop_browser = True
                    break

                # Check if .json file path is valid
                if not check_path_validity(stripped_path) or not stripped_path.endswith(
                    ".json"
                ):
                    print("Please enter a valid .json file path\n")
                    continue

                try:
                    with open(os.path.abspath(stripped_path), "r") as f:
                        new_tags = json.load(f)
                        valid_path = True

                except (
                    FileNotFoundError,
                    IOError,
                    OSError,
                    json.decoder.JSONDecodeError,
                ):
                    print(
                        f"{stripped_path} does not exist or doesnt have the right permissions to read.\n"
                        f"Please enter a valid .json file path."
                    )
                    continue

            if not valid_path:
                continue

            if len(new_tags) == 0:
                print(
                    f"No data found in {stripped_path}. Please check the path of the file passed"
                )
                continue
            print(f"Displaying the tags imported from {stripped_path}")
            merge_tags(new_tags, imported_tags, True)

        elif re.compile(
            "^export[\s]+tags[\s]+(all|(?:\s*(\d+))+)$", flags=re.IGNORECASE
        ).match(user_input):
            input_lst = user_input.split()
            if input_lst[2].lower() == "all":
                valid_indexes = [i + 1 for i in range(len(tf_records))]
            else:
                valid_indexes = check_index_validity(
                    input_lst[1:], len(tf_records), "display"
                )
                if len(valid_indexes) == 0:
                    continue

            tags_to_dump = {}
            for idx in valid_indexes:
                tfr_path = tf_records[idx - 1][1]
                tags_per_tfrecords.get(os.path.basename(tfr_path), {}),
                imported_tags.get(os.path.basename(tfr_path), {})
                valid_response = False
                print(
                    f"Which tags do you want to export from {os.path.basename(tfr_path)}?:\n"
                    "1. `Imported Tags` --> Tags imported from .json files.\n"
                    f"2. `Tags Added` --> Tags added by you.\n"
                    f"3. `Both Merged Together` --> Tags added by you and tags imported merged together.\n"
                    "Choose your response by entering 1, 2 or 3.\n"
                )
                while not valid_response:
                    try:
                        response = input("\nResponse: ")
                        stripped_response = response.strip()
                    except EOFError:
                        print("Raised EOF. Attempting to exit browser.")
                        stop_browser = True
                        break
                    if re.compile("^[1-3]$", re.IGNORECASE).match(stripped_response):
                        if stripped_response == "1":
                            tags_to_dump.update(
                                {
                                    os.path.basename(tfr_path): imported_tags.get(
                                        os.path.basename(tfr_path), {}
                                    )
                                }
                            )
                        elif stripped_response == "2":
                            tags_to_dump.update(
                                {
                                    os.path.basename(tfr_path): tags_per_tfrecords.get(
                                        os.path.basename(tfr_path), {}
                                    )
                                }
                            )
                        else:
                            tags_to_dump.update(
                                {
                                    os.path.basename(tfr_path): tags_per_tfrecords.get(
                                        os.path.basename(tfr_path), {}
                                    )
                                }
                            )
                            scenario_imported_tags = {
                                os.path.basename(tfr_path): imported_tags.get(
                                    os.path.basename(tfr_path), {}
                                )
                            }
                            merge_tags(scenario_imported_tags, tags_to_dump)
                        valid_response = True
                    else:
                        print(
                            "Invalid Response. Please choose your response by entering 1, 2, or 3.\n"
                        )

                if not valid_response:
                    break

            if stop_browser:
                continue

            valid_path = False
            print(
                "Enter the path to .json file to which you want to export the tags to?:\n"
            )
            while not valid_path:
                try:
                    response = input("\nEnter Path: ")
                    stripped_path = response.strip("[ \"']")
                except EOFError:
                    print("Raised EOF. Attempting to exit browser.")
                    stop_browser = True
                    break

                # Check if .json file path is valid
                if not check_path_validity(stripped_path) or not stripped_path.endswith(
                    ".json"
                ):
                    print("Please enter a valid .json file path\n")
                    continue

                try:
                    with open(os.path.abspath(stripped_path), "w") as f:
                        json.dump(tags_to_dump, f, ensure_ascii=False, indent=4)
                        print(f"Dumped the tags at {stripped_path}")
                        valid_path = True
                except (
                    FileNotFoundError,
                    IOError,
                    OSError,
                    json.decoder.JSONDecodeError,
                ):
                    print(
                        f"{stripped_path} is not valid json file or doesnt have the right permissions to write to this file.\n"
                        f"Please enter a valid .json path."
                    )
                    continue

            if not valid_path:
                continue

        elif re.compile("^explore[\s]+[\d]+$", flags=re.IGNORECASE).match(user_input):
            input_lst = user_input.split()
            valid_indexes = check_index_validity(
                [input_lst[1]], len(tf_records), "explore"
            )
            if len(valid_indexes) == 0:
                continue
            tfr_path = tf_records[valid_indexes[0] - 1][1]
            if scenarios_per_tfrecords[tfr_path] is None:
                (
                    scenarios_per_tfrecords[tfr_path],
                    tags_per_tfrecords[os.path.basename(tfr_path)],
                ) = get_scenario_and_tag_dict(tfr_path)
            stop_browser = explore_tf_record(
                tfr_path,
                scenarios_per_tfrecords[tfr_path],
                tags_per_tfrecords[os.path.basename(tfr_path)],
                imported_tags,
                default_target_path,
            )
            if not stop_browser:
                display_tf_records(tf_records)
            print_commands = True

        elif user_input.lower() == "exit":
            stop_browser = True

        else:
            print(
                "Invalid command. Please enter a valid command. See command formats above"
            )
    print(
        "If you exported any scenarios, you can build them using the command `scl scenario build <target_base_path>`.\n"
        "Have a look at README.md at the root level of this repo for more info on how to build scenarios."
    )
    print("Exiting the Browser")


def explore_tf_record(
    tfrecord: str,
    scenario_dict,
    tfrecord_tags,
    imported_tags,
    default_target_path: Optional[str] = None,
) -> bool:
    scenario_ids = display_scenarios_in_tfrecord(
        tfrecord,
        scenario_dict,
        tfrecord_tags,
        imported_tags.get(os.path.basename(tfrecord), {}),
    )
    stop_exploring = False
    print_commands = True
    while not stop_exploring:
        if print_commands:
            print("\n")
            print(
                f"{os.path.basename(tfrecord)} TfRecord Browser.\n"
                f"You can use the following commands to further explore these scenarios:\n"
                "1. `export all/<indexes>` --> Export the scenarios at these indexes of the table to a target path\n"
                f"                             The indexes should be an integer between 1 and {len(scenario_ids)} separated by space"
                f"                             Optionally you can use 'export all` command to export all the scenarios.\n"
                "3. `preview all` --> Plot and dump the images of the map of all scenarios in this tf_record to a target path.\n"
                "4. `preview <indexes>` --> Plot and display the maps of these scenario at these index of the table.\n"
                f"                          The indexes should be an integer between 1 and {len(scenario_ids)} and should be separated by space.\n"
                f"5. `tag all/<indexes>` or `tag imported all/<indexes>` --> Tag the scenario at these indexes of the table or all with tags mentioned.\n"
                f"                                                           Optionally if you call with `tag imported` then the tags for these scenarios will be added to imported tag list.\n"
                f"                                                           If indexes, then they be integers between 1 and {len(scenario_ids)} and should be separated by space.\n"
                f"6. `untag all/<indexes>` or `untag imported all/<indexes>` --> Untag the scenario at theses index of the table or all with tags mentioned.\n"
                f"                                                               Optionally if you call with `tag imported` then the tags for these scenarios will be removed from imported tag list.\n"
                f"                                                               If indexes, then they be integers between 1 and {len(scenario_ids)} and should be separated by space.\n"
                f"7. `select <index>` --> Select and explore further the scenario at this index of the table.\n"
                f"                        The index should be an integer between 1 and {len(scenario_ids)}\n"
                "8. `animate all` --> Plot and dump the animations the trajectories of objects on map of all scenarios in this tf_record to a target path.\n"
                f"9. `animate <indexes>` --> Plot the map and animate the trajectories of objects of scenario at this index of the table.\n"
                f"                           The indexes should be an integer between 1 and {len(scenario_ids)} and should be separated by space.\n"
                "10. `go back` --> Go back to the tfrecords browser\n"
                "11. `exit` --> Exit the program\n"
            )
            print_commands = False

        print("\n")
        try:
            raw_input = input("Command: ")
            user_input = raw_input.strip()
        except EOFError:
            print("Raised EOF. Attempting to exit browser.")
            return True

        if re.compile("^export[\s]+(all|(?:\s*(\d+))+)", flags=re.IGNORECASE).match(
            user_input
        ):
            input_lst = user_input.split()
            if input_lst[1].lower() == "all":
                valid_indexes = [i + 1 for i in range(len(scenario_ids))]
            else:
                valid_indexes = check_index_validity(
                    input_lst[1:], len(scenario_ids), "export"
                )
                if len(valid_indexes) == 0:
                    continue
            target_base_path = None
            valid_path = False
            if default_target_path is not None:
                print(
                    f"Which path do you want to export scenarios to?:\n"
                    "1. Default Target Path.\n"
                    f"2. Custom Target Path.\n"
                    "Choose your response by entering 1,or 2.\n"
                )
                valid_response = False
                while not valid_response:
                    try:
                        response = input("\nResponse: ")
                        stripped_response = response.strip()
                    except EOFError:
                        print("Raised EOF. Attempting to exit browser.")
                        return True
                    if re.compile("^[1-2]$", re.IGNORECASE).match(stripped_response):
                        if stripped_response == "1":
                            target_base_path = default_target_path
                            valid_path = True
                        valid_response = True
                    else:
                        print(
                            "Invalid Response. Please choose your response by entering 1 or 2.\n"
                        )

            while not valid_path:
                print(
                    "Enter the path to directory to which you want to export the scenarios?:\n"
                )
                try:
                    response = input("\nEnter Path: ")
                    stripped_path = response.strip("[ \"']")
                except EOFError:
                    print("Raised EOF. Attempting to exit browser.")
                    return True

                # Check if directory path is valid
                if not check_path_validity(stripped_path):
                    print("Please enter a valid directory path\n")
                    continue
                target_base_path = stripped_path
                valid_path = True

            # Try exporting the scenario
            exported = False
            for idx in valid_indexes:
                exported = (
                    export_scenario(target_base_path, tfrecord, scenario_ids[idx - 1])
                    or exported
                )
            if exported:
                print(
                    f"\nYou can build these scenarios exported using the command `scl scenario build-all {target_base_path}`"
                )
            display_scenarios_in_tfrecord(
                tfrecord,
                scenario_dict,
                tfrecord_tags,
                imported_tags.get(os.path.basename(tfrecord), {}),
            )
            print_commands = True

        elif re.compile("^preview[\s]+all$", flags=re.IGNORECASE).match(user_input):
            print(
                "Enter the path to directory to which you want to dump the images of the maps of scenarios?:\n"
            )
            valid_path = False
            target_base_path = None
            while not valid_path:
                try:
                    response = input("\nEnter Path: ")
                    stripped_path = response.strip("[ \"']")
                except EOFError:
                    print("Raised EOF. Attempting to exit browser.")
                    return True

                # Check if directory path is valid
                if not check_path_validity(stripped_path):
                    print("Please enter a valid directory path\n")
                    continue
                target_base_path = stripped_path
                valid_path = True

            # Dump all the scenario plots of this tfrecord file to this target base path
            print(
                f"Plotting and dumping all the scenario maps in {tfrecord} tfrecord file"
            )
            dump_plots(target_base_path, scenario_dict)
            display_scenarios_in_tfrecord(
                tfrecord,
                scenario_dict,
                tfrecord_tags,
                imported_tags.get(os.path.basename(tfrecord), {}),
            )
            print_commands = True

        elif re.compile("^preview[\s]+(?:\s*(\d+))+$", flags=re.IGNORECASE).match(
            user_input
        ):
            input_lst = user_input.split()

            # Check if index passed is valid
            valid_indexes = check_index_validity(
                input_lst[1:], len(scenario_ids), "preview"
            )
            if len(valid_indexes) == 0:
                continue

            # Plot the maps of these scenarios
            scenarios_to_plot = []
            for i in range(len(valid_indexes)):
                scenario_idx = scenario_ids[valid_indexes[i] - 1]
                if scenario_dict[scenario_idx][1] is None:
                    scenario_dict[scenario_idx][1] = get_map_features_for_scenario(
                        scenario_dict[scenario_idx][0]
                    )
                scenarios_to_plot.append(scenario_dict[scenario_idx])

            plot_scenarios(scenarios_to_plot, False)

        elif re.compile("^animate[\s]+all$", flags=re.IGNORECASE).match(user_input):
            print(
                "Enter the path to directory to which you want to dump the animations of the track objects of scenarios?:\n"
            )
            valid_path = False
            target_base_path = None
            while not valid_path:
                try:
                    response = input("\nEnter Path: ")
                    stripped_path = response.strip("[ \"']")
                except EOFError:
                    print("Raised EOF. Attempting to exit browser.")
                    return True

                # Check if directory path is valid
                if not check_path_validity(stripped_path):
                    print("Please enter a valid directory path\n")
                    continue
                target_base_path = stripped_path
                valid_path = True

            # Dump all the scenario plots of this tfrecord file to this target base path
            print(
                f"Plotting and dumping all the scenarios animations in {tfrecord} tfrecord file"
            )
            dump_plots(target_base_path, scenario_dict, animate=True)
            display_scenarios_in_tfrecord(
                tfrecord,
                scenario_dict,
                tfrecord_tags,
                imported_tags.get(os.path.basename(tfrecord), {}),
            )
            print_commands = True

        elif re.compile("^animate[\s]+(?:\s*(\d+))+$", flags=re.IGNORECASE).match(
            user_input
        ):
            input_lst = user_input.split()

            # Check if index passed is valid
            valid_indexes = check_index_validity(
                input_lst[1:], len(scenario_ids), "animate"
            )
            if len(valid_indexes) == 0:
                continue

            # Animate the maps of these scenarios
            scenarios_to_animate = []
            for i in range(len(valid_indexes)):
                scenario_idx = scenario_ids[valid_indexes[i] - 1]
                if scenario_dict[scenario_idx][1] is None:
                    scenario_dict[scenario_idx][1] = get_map_features_for_scenario(
                        scenario_dict[scenario_idx][0]
                    )

                if scenario_dict[scenario_idx][2] is None:
                    scenario_dict[scenario_idx][2] = get_trajectory_data(
                        scenario_dict[scenario_idx][0]
                    )
                scenarios_to_animate.append(scenario_dict[scenario_idx])

            plot_scenarios(scenarios_to_animate, True)

        elif re.compile(
            "^tag([\s]+imported)?[\s]+(all|(?:\s*(\d+))+)$", flags=re.IGNORECASE
        ).match(user_input):
            input_lst = user_input.lower().split()
            imported = True if "imported" in input_lst else False
            if "all" in input_lst:
                valid_indexes = [i + 1 for i in range(len(scenario_ids))]
            else:
                if imported:
                    indexes_input = input_lst[2:]
                else:
                    indexes_input = input_lst[1:]

                # Check if index passed is valid
                valid_indexes = check_index_validity(
                    indexes_input, len(scenario_ids), "tag"
                )
            if len(valid_indexes) == 0:
                continue
            print(
                "What Tags do you want to add?\n"
                "Your response should have tags that are alphanumerical and can have special characters but need to be separated by Comma.\n"
            )
            valid_response = False
            stripped_response = None
            while not valid_response:
                try:
                    response = input("\nResponse: ")
                    stripped_response = response.strip()
                except EOFError:
                    print("Raised EOF. Attempting to exit browser.")
                    return True
                if stripped_response == "":
                    print(
                        "Invalid response. Your response should have tags that are alphanumerical and can have special characters but need to be separated by Comma."
                    )
                    continue
                else:
                    valid_response = True

            tags = [tag.strip() for tag in stripped_response.lower().split(",")]
            if imported:
                if os.path.basename(tfrecord) in imported_tags:
                    for i in range(len(valid_indexes)):
                        scenario_idx = scenario_ids[valid_indexes[i] - 1]
                        if scenario_idx in imported_tags[os.path.basename(tfrecord)]:
                            imported_tags[os.path.basename(tfrecord)][
                                scenario_idx
                            ].extend(
                                [
                                    tag
                                    for tag in tags
                                    if tag
                                    not in imported_tags[os.path.basename(tfrecord)][
                                        scenario_idx
                                    ]
                                ]
                            )
                        else:
                            imported_tags[os.path.basename(tfrecord)][
                                scenario_idx
                            ] = tags
                else:
                    imported_tags[os.path.basename(tfrecord)] = {
                        scenario_ids[valid_indexes[i] - 1]: tags
                        for i in range(len(valid_indexes))
                    }
                print("Tags added to `Imported Tags` list")
            else:
                for i in range(len(valid_indexes)):
                    scenario_idx = scenario_ids[valid_indexes[i] - 1]
                    tfrecord_tags[scenario_idx].extend(
                        [tag for tag in tags if tag not in tfrecord_tags[scenario_idx]]
                    )
                print("Tags added to `Tags Added` list")
            display_scenarios_in_tfrecord(
                tfrecord,
                scenario_dict,
                tfrecord_tags,
                imported_tags.get(os.path.basename(tfrecord), {}),
            )
            print_commands = True

        elif re.compile(
            "^untag([\s]+imported)?[\s]+(all|(?:\s*(\d+))+)$", flags=re.IGNORECASE
        ).match(user_input):
            input_lst = user_input.lower().split()
            imported = True if input_lst[1] == "imported" else False
            if "all" in input_lst:
                valid_indexes = [i + 1 for i in range(len(scenario_ids))]
            else:
                if imported:
                    indexes_input = input_lst[2:]
                else:
                    indexes_input = input_lst[1:]

                # Check if index passed is valid
                valid_indexes = check_index_validity(
                    indexes_input, len(scenario_ids), "tag"
                )
            if len(valid_indexes) == 0:
                continue
            print(
                "What Tags do you want to remove? Tags that dont exist won't be removed.\n"
                "Your response should have tags that are alphanumerical and can have special characters but need to be separated by Comma.\n"
                "Optionally you can respond `remove all`, to remove all tags from these scenarios."
            )
            valid_response = False
            stripped_response = None
            while not valid_response:
                try:
                    response = input("\nResponse: ")
                    stripped_response = response.strip()
                except EOFError:
                    print("Raised EOF. Attempting to exit browser.")
                    return True
                if stripped_response == "":
                    print(
                        "Invalid response. Your response should have tags that are alphanumerical and can have special characters but need to be separated by Comma."
                    )
                    continue
                else:
                    valid_response = True

            tags = [tag.strip() for tag in stripped_response.lower().split(",")]

            # if response is remove all, remove all the tags from the scenarios mentioned
            remove_all = False
            if "remove all" in tags:
                remove_all = True

            if imported:
                if os.path.basename(tfrecord) in imported_tags:
                    for i in range(len(valid_indexes)):
                        scenario_idx = scenario_ids[valid_indexes[i] - 1]
                        if scenario_idx in imported_tags[os.path.basename(tfrecord)]:
                            if remove_all:
                                imported_tags[os.path.basename(tfrecord)][
                                    scenario_idx
                                ] = []
                            else:
                                new_tags = []
                                for tag in imported_tags[os.path.basename(tfrecord)][
                                    scenario_idx
                                ]:
                                    if tag not in tags:
                                        new_tags.append(tag)
                                imported_tags[os.path.basename(tfrecord)][
                                    scenario_idx
                                ] = new_tags
                                print(
                                    f"Tags removed from `Imported Tags` list of {scenario_idx}"
                                )
                        else:
                            print(f"no imported tags for {scenario_idx}")
                else:
                    print(
                        f"No tags for {os.path.basename(tfrecord)} in imported tags list"
                    )
                    continue

            else:
                for i in range(len(valid_indexes)):
                    scenario_idx = scenario_ids[valid_indexes[i] - 1]
                    if len(tfrecord_tags[scenario_idx]) == 0:
                        print(f"No tags added for {scenario_idx} that can be removed")
                    else:
                        new_tags = []
                        for tag in tfrecord_tags[scenario_idx]:
                            if tag not in tags:
                                new_tags.append(tag)
                        tfrecord_tags[scenario_idx] = new_tags
                        print(f"Tags removed from `Tags Added` list of {scenario_idx}")
            display_scenarios_in_tfrecord(
                tfrecord,
                scenario_dict,
                tfrecord_tags,
                imported_tags.get(os.path.basename(tfrecord), {}),
            )
            print_commands = True

        elif re.compile("^select[\s]+[\d]+$", flags=re.IGNORECASE).match(user_input):
            input_lst = user_input.split()

            # Check if index passed is valid
            valid_indexes = check_index_validity(
                input_lst[1:], len(scenario_ids), "select"
            )
            if len(valid_indexes) == 0:
                continue

            # Explore further the scenario at this index
            scenario_id = scenario_ids[valid_indexes[0] - 1]
            if scenario_dict[scenario_id][1] is None:
                scenario_dict[scenario_id][1] = get_map_features_for_scenario(
                    scenario_dict[scenario_id][0]
                )
            if scenario_dict[scenario_id][2] is None:
                scenario_dict[scenario_id][2] = get_trajectory_data(
                    scenario_dict[scenario_id][0]
                )
            exit_browser = explore_scenario(
                tfrecord,
                scenario_dict[scenario_id],
                tfrecord_tags[scenario_id],
                imported_tags,
                default_target_path,
            )
            if exit_browser:
                return True
            display_scenarios_in_tfrecord(
                tfrecord,
                scenario_dict,
                tfrecord_tags,
                imported_tags.get(os.path.basename(tfrecord), {}),
            )
            print_commands = True

        elif re.compile("^go[\s]+back$", flags=re.IGNORECASE).match(user_input):
            stop_exploring = True
            print("Going back to the tfRecords browser")
            continue

        elif user_input.lower() == "exit":
            return True
        else:
            print(
                "Invalid command. Please enter a valid command. See command formats above"
            )
    return False


def explore_scenario(
    tfrecord_file_path: str,
    scenario_info,
    scenario_tags,
    imported_tags,
    default_target_path: Optional[str],
) -> bool:

    scenario = scenario_info[0]
    imported_tfrecord_tags = imported_tags.get(os.path.basename(tfrecord_file_path), {})
    imported_scenario_tags = imported_tfrecord_tags.get(scenario.scenario_id, [])
    scenario_map_features = scenario_info[1]
    trajectories = scenario_info[2]
    scenario_data = [
        scenario.scenario_id,
        len(scenario.timestamps_seconds),
        len(scenario.tracks),
        len(scenario.dynamic_map_states),
        len(scenario.objects_of_interest),
        scenario_tags,
        imported_scenario_tags,
    ]

    def display_scenario_data_info():
        print("\n")
        print("-----------------------------------------------")
        print(f"Scenario {scenario.scenario_id}:\n")
        print(
            tabulate(
                [scenario_data],
                headers=[
                    "Scenario ID",
                    "Timestamps",
                    "Track Objects",
                    "Traffic Light States",
                    "Objects of Interest",
                    "Tags Added",
                    "Tags Imported",
                ],
            )
        )

    display_scenario_data_info()

    map_features = [
        len(scenario_map_features["lane"]),
        len(scenario_map_features["road_line"]),
        len(scenario_map_features["road_edge"]),
        len(scenario_map_features["stop_sign"]),
        len(scenario_map_features["crosswalk"]),
        len(scenario_map_features["speed_bump"]),
    ]
    print(f"\n\nScenario {scenario.scenario_id} map data:\n")
    print(
        tabulate(
            [map_features],
            headers=[
                "Lanes",
                "Road Lines",
                "Road Edges",
                "Stop Signs",
                "Crosswalks",
                "Speed Bumps",
            ],
        )
    )
    print("\n\nLane Ids: ", [lane[1] for lane in scenario_map_features["lane"]])
    print(
        "\nRoad Line Ids: ",
        [road_line[1] for road_line in scenario_map_features["road_line"]],
    )
    print(
        "\nRoad Edge Ids: ",
        [road_edge[1] for road_edge in scenario_map_features["road_edge"]],
    )
    print(
        "\nStop Sign Ids: ",
        [stop_sign[1] for stop_sign in scenario_map_features["stop_sign"]],
    )
    print(
        "\nCrosswalk Ids: ",
        [crosswalk[1] for crosswalk in scenario_map_features["crosswalk"]],
    )
    print(
        "\nSpeed Bumps Ids: ",
        [speed_bump[1] for speed_bump in scenario_map_features["speed_bump"]],
    )
    print("\n-----------------------------------------------")
    ego, cars, pedestrian, cyclist, others = get_object_type_count(trajectories)
    print("Trajectory Data")
    trajectory_data = [
        scenario.scenario_id,
        len(cars) + 1,
        len(pedestrian),
        len(cyclist),
        len(others),
    ]
    print(
        tabulate(
            [trajectory_data],
            headers=[
                "Scenario ID",
                "Cars",
                "Pedestrians",
                "Cyclists",
                "Others",
            ],
        )
    )
    print("\n\nTrack Object Ids: ")
    print("\nEgo Id: ", ego)
    print("\nCar Ids: ", cars)
    print("\nPedestrian Ids: ", pedestrian)
    print("\nCyclist Ids: ", cyclist)
    print("\nOther Ids: ", others)

    print(
        f"\n\nScenario {scenario.scenario_id}.\n"
        "You can use the following commands to further this scenario:\n"
        f"1. `export` --> Export the scenario to a target base path is passed.\n"
        f"2. `preview` or `preview <feature_ids>` --> Plot and display the map of the scenario with the feature ids highlighted in Blue if passed.\n"
        f"                                            The feature ids need to be separated by space, be numbers from the map feature ids mentioned above and will not be highlighted if they dont exist.\n"
        "3. `animate` or `animate <track_ids> --> Animate the trajectories of track objects on the map of this scenario with the track ids highlighted in Red if passed.\n"
        f"                                        The track ids need to be separated by space, be numbers from the track object ids mentioned above and will not be highlighted if they dont exist.\n"
        f"4. `tag` or `tag imported` --> Tag the scenario with tags mentioned.\n"
        "                                Optionally if you call with `tag imported` then the tags will be added to imported tag list.\n"
        f"5. `untag` or `untag imported` --> Untag the scenario with tags mentioned.\n"
        "                                    Optionally if you call with `untag imported` then the tags will be removed to imported tag list.\n"
        "6. `go back` --> Go back to this scenario's tfrecord browser.\n"
        "7. `exit` --> Exit the program\n"
    )
    stop_exploring = False
    while not stop_exploring:
        try:
            raw_input = input("\nCommand: ")
            user_input = raw_input.strip()
        except EOFError:
            print("Raised EOF. Attempting to exit browser.")
            return True
        if user_input.lower() == "export":
            target_base_path = None
            valid_path = False
            if default_target_path is not None:
                print(
                    f"Which path do you want to export scenario to?:\n"
                    "1. Default Target Path.\n"
                    f"2. Custom Target Path.\n"
                    "Choose your response by entering 1 or 2.\n"
                )
                valid_response = False
                while not valid_response:
                    try:
                        response = input("\nResponse: ")
                        stripped_response = response.strip()
                    except EOFError:
                        print("Raised EOF. Attempting to exit browser.")
                        return True
                    if re.compile("^[1-2]$", re.IGNORECASE).match(stripped_response):
                        if stripped_response == "1":
                            target_base_path = default_target_path
                            valid_path = True
                        valid_response = True
                    else:
                        print(
                            "Invalid Response. Please choose your response by entering 1 or 2.\n"
                        )

            while not valid_path:
                print(
                    "Enter the path to directory to which you want to export the scenarios?:\n"
                )
                try:
                    response = input("\nEnter Path: ")
                    stripped_path = response.strip("[ \"']")
                except EOFError:
                    print("Raised EOF. Attempting to exit browser.")
                    return True

                # Check if directory path is valid
                if not check_path_validity(stripped_path):
                    print("Please enter a valid directory path\n")
                    continue
                target_base_path = stripped_path
                valid_path = True

            # Try exporting the scenario to the target_base_path
            export_scenario(target_base_path, tfrecord_file_path, scenario.scenario_id)
            print(
                f"\nYou can build the scenario exported using the command `scl scenario build {target_base_path}`"
            )

        elif user_input.lower() == "preview" or re.compile(
            "^preview[\s]+(?:\s*(\d+))+$", flags=re.IGNORECASE
        ).match(user_input):
            input_lst = user_input.split()
            if len(input_lst) == 1:
                # Plot this scenario
                plot_scenarios([scenario_info], False)
            else:
                plot_scenarios([scenario_info], False, input_lst[1:])

        elif user_input.lower() == "animate" or re.compile(
            "^animate[\s]+(?:\s*(\d+))+?$", flags=re.IGNORECASE
        ).match(user_input):
            # Animate this scenario
            input_lst = user_input.split()
            if len(input_lst) == 1:
                # Plot this scenario
                plot_scenarios([scenario_info], True)
            else:
                plot_scenarios([scenario_info], True, input_lst[1:])

        elif re.compile("^tag([\s]+imported)?$", flags=re.IGNORECASE).match(user_input):
            print(
                "What Tags do you want to add?\n"
                "Your response should have tags that are alphanumerical and can have special characters but need to be separated by Comma."
            )
            input_lst = user_input.lower().split()
            imported = True if "imported" in input_lst else False
            valid_response = False
            stripped_response = None
            while not valid_response:
                try:
                    response = input("\nResponse: ")
                    stripped_response = response.strip()
                except EOFError:
                    print("Raised EOF. Attempting to exit browser.")
                    return True
                if stripped_response == "":
                    print(
                        "Invalid response. Your response should have tags that are alphanumerical and can have special characters but need to be separated by Comma."
                    )
                    continue
                else:
                    valid_response = True

            tags = [tag.strip() for tag in stripped_response.lower().split(",")]
            if imported:
                if os.path.basename(tfrecord_file_path) in imported_tags:
                    if (
                        scenario.scenario_id
                        in imported_tags[os.path.basename(tfrecord_file_path)]
                    ):
                        imported_tags[os.path.basename(tfrecord_file_path)][
                            scenario.scenario_id
                        ].extend(
                            [
                                tag
                                for tag in tags
                                if tag
                                not in imported_tags[
                                    os.path.basename(tfrecord_file_path)
                                ][scenario.scenario_id]
                            ]
                        )
                    else:
                        imported_tags[os.path.basename(tfrecord_file_path)][
                            scenario.scenario_id
                        ] = tags
                else:
                    imported_tags[os.path.basename(tfrecord_file_path)] = {
                        scenario.scenario_id: tags
                    }
                print("Tags added to `Imported Tags` list")
            else:
                scenario_tags.extend([tag for tag in tags if tag not in scenario_tags])
                print("Tags added to `Tags Added` list")
            display_scenario_data_info()

        elif re.compile("^untag([\s]+imported)?$", flags=re.IGNORECASE).match(
            user_input
        ):
            print(
                "What Tags do you want to remove?. Tags that don't exist wont be removed.\n"
                "Your response should have tags that are alphanumerical and can have special characters but need to be separated by Comma."
            )
            input_lst = user_input.lower().split()
            imported = True if "imported" in input_lst else False
            valid_response = False
            stripped_response = None
            while not valid_response:
                try:
                    response = input("\nResponse: ")
                    stripped_response = response.strip()
                except EOFError:
                    print("Raised EOF. Attempting to exit browser.")
                    return True
                if stripped_response == "":
                    print(
                        "Invalid response. Your response should have tags that are alphanumerical and can have special characters but need to be separated by Comma."
                    )
                else:
                    valid_response = True

            tags = [tag.strip() for tag in stripped_response.lower().split(",")]
            scenario_idx = scenario.scenario_id
            if imported:
                if os.path.basename(tfrecord_file_path) in imported_tags:
                    if (
                        scenario_idx
                        in imported_tags[os.path.basename(tfrecord_file_path)]
                    ):
                        new_tags = []
                        for tag in imported_tags[os.path.basename(tfrecord_file_path)][
                            scenario_idx
                        ]:
                            if tag not in tags:
                                new_tags.append(tag)
                        imported_tags[os.path.basename(tfrecord_file_path)][
                            scenario_idx
                        ] = new_tags
                    else:
                        print(f"no imported tags for {scenario_idx}")
                        continue
                else:
                    print(
                        f"No tags for {os.path.basename(tfrecord_file_path)} in imported tags list"
                    )
                    continue
                print("Tags removed from `Imported Tags` list")
            else:
                if len(scenario_tags) == 0:
                    print(f"No tags added for {scenario_idx} that can be removed")
                else:
                    new_tags = []
                    for tag in scenario_tags:
                        if tag not in tags:
                            new_tags.append(tag)
                    scenario_tags = new_tags
                    print("Tags removed from `Tags Added` list")
            display_scenario_data_info()

        elif re.compile("^go[\s]+back$", flags=re.IGNORECASE).match(user_input):
            stop_exploring = True
            print("Going back to the tfRecord Explorer")
            continue

        elif user_input.lower() == "exit":
            return True
        else:
            print(
                "Invalid command. Please enter a valid command. See command formats above"
            )
    return False


if __name__ == "__main__":
    import warnings
    import readline

    warnings.filterwarnings("ignore", category=DeprecationWarning)

    parser = argparse.ArgumentParser(
        prog="waymo_utility.py",
        description="Text based TfRecords Browser.",
    )
    parser.add_argument(
        "files",
        help="A list of TFRecord file/folder paths. Each element can be either the path to "
        "tfrecord file or a directory of tfrecord files to browse from.",
        type=str,
        nargs="+",
    )
    parser.add_argument(
        "--target-base-path",
        help="Default target base path to export scenarios to",
        type=str,
        default=None,
    )
    args = parser.parse_args()
    valid_tf_paths = []
    if args.target_base_path is not None:
        if not os.path.exists(os.path.abspath(args.target_base_path)):
            print(
                f"Default Target Base Path {args.target_base_path} does not exist.\n"
                f"Please make sure Default Target Base path passed is valid and it exists if you pass it."
            )
            exit()
    for tf_path in args.files:
        if not os.path.exists(os.path.abspath(tf_path)):
            print(
                f"Path {args.file} does not exist and hence wont be browsed.\n"
                f"Please make sure path passed is valid and it exists."
            )
        else:
            valid_tf_paths.append(os.path.abspath(tf_path))
    if not valid_tf_paths:
        print("No valid paths passed. Make sure all paths passed exist and are valid.")
    else:
        tfrecords_browser(
            valid_tf_paths,
            os.path.abspath(args.target_base_path) if args.target_base_path else None,
        )
