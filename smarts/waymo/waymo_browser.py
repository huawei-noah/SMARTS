# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
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

# Text based Waymo Dataset Browser.
import argparse
import copy
import json
import os
import re
import struct
from itertools import product
from multiprocessing import cpu_count
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.lines import Line2D

from smarts.waymo import waymo_utils

try:
    import readline

    from pathos.multiprocessing import ProcessingPool as Pool
    from tabulate import tabulate
    from waymo_open_dataset.protos import scenario_pb2
except (ModuleNotFoundError, ImportError):
    import sys
    from collections import namedtuple
    from typing import Any

    scenario_pb2 = namedtuple("scenario_pb2", "Scenario")(Any)
    print(sys.exc_info())
    print(
        "Unable to run Waymo utility. To enable, pip install the missing dependencies.\n"
        "pip install pathos==0.2.8 tabulate>=0.8.10 waymo-open-dataset-tf-2-4-0"
    )
    if __name__ == "__main__":
        exit()


def read_tfrecord_file(path: str) -> Generator[bytes, None, None]:
    """Iterate over the records in a TFRecord file and return the bytes of each record.

    path: The path to the TFRecord file
    """
    with open(path, "rb") as f:
        while True:
            length_bytes = f.read(8)
            if len(length_bytes) != 8:
                return
            record_len = int(struct.unpack("Q", length_bytes)[0])
            _ = f.read(4)  # masked_crc32_of_length (ignore)
            record_data = f.read(record_len)
            _ = f.read(4)  # masked_crc32_of_data (ignore)
            yield record_data


def convert_polyline(polyline) -> Tuple[List[float], List[float]]:
    """Convert a MapFeature.MapPoint Waymo proto object to a tuple of x and y coordinates

    polyline: MapFeature.MapPoint Waymo proto object
    """
    xs, ys = [], []
    for p in polyline:
        xs.append(p.x)
        ys.append(p.y)
    return xs, ys


def get_map_features_for_scenario(scenario: scenario_pb2.Scenario) -> Dict:
    """Extract all map features from Scenario object

    scenario: scenario_pb2.Scenario waymo proto object
    """
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


def get_object_type_count(
    trajectories: Dict,
) -> Tuple[Optional[int], List[int], List[int], List[int], List[int]]:
    """Get count of all types of Track objects

    trajectories: Dictionary containing trajectory info for all track objects
    """
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


def get_trajectory_data(waymo_scenario: scenario_pb2.Scenario) -> Dict:
    """Get count of all types of Track objects

    trajectories: Dictionary containing trajectory info for all track objects
    """

    def generate_trajectory_rows(
        scenario: scenario_pb2.Scenario,
    ) -> Generator[Dict, None, None]:
        """Generator to yield trajectory data of every track object at every timestep"""
        for i in range(len(scenario.tracks)):
            vehicle_id = scenario.tracks[i].id
            num_steps = len(scenario.timestamps_seconds)
            # First pass -- extract data and yield trajectory at every timestep
            for j in range(num_steps):
                obj_state = scenario.tracks[i].states[j]
                row = dict()
                row["vehicle_id"] = vehicle_id
                if not obj_state.valid:
                    row["position_x"] = None
                    row["position_y"] = None
                row["type"] = scenario.tracks[i].object_type
                row["is_ego_vehicle"] = 1 if i == scenario.sdc_track_index else 0
                row["position_x"] = obj_state.center_x
                row["position_y"] = obj_state.center_y
                yield row

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


def plot_map_features(map_features: Dict, feature_ids: List[str]) -> List[Line2D]:
    """Plot the map features with some feature_ids highlighted and return extended legend handles"""
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


def plot_trajectories(
    trajectories: Dict, objects_of_interest: List[int], track_ids: List[str]
) -> Tuple:
    """Plot and animate the trajectories of track ids on map of scenario with some track_ids highlighted"""
    handles = []
    max_len = 0
    data, points = [], []

    # Need to plot something initially to get handles to the point objects,
    # so just use a valid point from the first trajectory
    first_traj = list(trajectories.values())[0]
    ind = min(range(len(first_traj[1])), key=first_traj[1].__getitem__)
    x0 = first_traj[0][ind]
    y0 = first_traj[1][ind]
    for k, v in trajectories.items():
        xs, ys = v[0], v[1]
        is_ego = v[2]
        object_type = v[3]
        if len(xs) > max_len:
            max_len = len(xs)
        if is_ego:
            (point,) = plt.plot(x0, y0, "c^")
        elif object_type == 1:
            if str(k) in track_ids:
                (point,) = plt.plot(x0, y0, "r^")
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
            elif int(k) in objects_of_interest:
                (point,) = plt.plot(x0, y0, "g^")
                handles.append(
                    Line2D(
                        [],
                        [],
                        color="green",
                        marker="^",
                        linestyle="None",
                        markersize=5,
                        label=f"Interactive Car {k}",
                    )
                )
            else:
                (point,) = plt.plot(x0, y0, "k^")
        elif object_type == 2:
            if str(k) in track_ids:
                (point,) = plt.plot(x0, y0, "rd")
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
            elif int(k) in objects_of_interest:
                (point,) = plt.plot(x0, y0, "gd")
                handles.append(
                    Line2D(
                        [],
                        [],
                        color="green",
                        marker="d",
                        linestyle="None",
                        markersize=5,
                        label=f"Interactive Pedestrian {k}",
                    )
                )
            else:
                (point,) = plt.plot(x0, y0, "md")
        elif object_type == 3:
            if str(k) in track_ids:
                (point,) = plt.plot(x0, y0, "r*")
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
            elif int(k) in objects_of_interest:
                (point,) = plt.plot(x0, y0, "g*")
                handles.append(
                    Line2D(
                        [],
                        [],
                        color="green",
                        marker="*",
                        linestyle="None",
                        markersize=5,
                        label=f"Interactive Cyclist {k}",
                    )
                )
            else:
                (point,) = plt.plot(x0, y0, "y*")
        else:
            if str(k) in track_ids:
                (point,) = plt.plot(x0, y0, "r8")
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
            elif int(k) in objects_of_interest:
                (point,) = plt.plot(x0, y0, "g8")
                handles.append(
                    Line2D(
                        [],
                        [],
                        color="green",
                        marker="8",
                        linestyle="None",
                        markersize=5,
                        label=f"Interactive Other {k}",
                    )
                )
            else:
                (point,) = plt.plot(x0, y0, "k8")
        data.append((xs, ys))
        points.append(point)

    return data, points, max_len, handles


def plot_scenario(
    scenario: Tuple[List, int],
    animate_trajectories: bool,
    f_ids: Optional[List[str]] = None,
):
    """Plot the map of scenario and optionally animate trajectories of track ids with f_ids highlighted"""
    scenario_info, fig_num = scenario
    # Get map feature data from map proto
    map_features = scenario_info[1]

    # Plot map
    fig = plt.figure()
    if animate_trajectories or not f_ids:
        highlighted_handles = plot_map_features(map_features, [])
    else:
        highlighted_handles = plot_map_features(map_features, f_ids)
    plt.title(f"Scenario {scenario_info[0].scenario_id}, idx {fig_num}")

    # Set Legend Handles
    all_handles = []
    all_handles.extend(waymo_utils.MAP_HANDLES + highlighted_handles)

    if animate_trajectories:
        # Plot Trajectories
        data, points, max_len, t_handles = plot_trajectories(
            scenario_info[2],
            scenario_info[0].objects_of_interest,
            f_ids if f_ids else [],
        )
        all_handles.extend(waymo_utils.TRAJECTORY_HANDLES + t_handles)

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
    plt.show()


def save_plot(
    scenario_info: Tuple[int, str],
    target_path: str,
    scenario_dict: Dict,
    animate: bool,
    filter_tags: Optional[Tuple] = None,
) -> bool:
    """Plot and save the map of scenario at the target_path optionally filtered with tags.
    If animate is true, .mp4 video player showing the animation of trajectories of track objects will be saved
    """
    idx, scenario_id = scenario_info
    if filter_tags:
        tags, filter_preview, tfrecord_tags, imported_tfrecord_tags = filter_tags
        if not filter_scenario(
            tfrecord_tags.get(scenario_id, []),
            imported_tfrecord_tags.get(scenario_id, []),
            (tags, filter_preview),
        ):
            return False

    scenario = scenario_dict[scenario_id][0]

    fig = plt.figure()
    mng = plt.get_current_fig_manager()
    mng.resize(1000, 1000)
    plt.title(f"Scenario {scenario_id}, index {idx}")

    # Plot map
    if scenario_dict[scenario_id][1] is None:
        scenario_dict[scenario_id][1] = get_map_features_for_scenario(scenario)
    plot_map_features(scenario_dict[scenario_id][1], [])
    all_handles = []
    all_handles.extend(waymo_utils.MAP_HANDLES)

    if animate:
        # Plot Trajectories
        if scenario_dict[scenario_id][2] is None:
            scenario_dict[scenario_id][2] = get_trajectory_data(scenario)
        data, points, max_len, _ = plot_trajectories(
            scenario_dict[scenario_id][2],
            scenario_dict[scenario_id][0].objects_of_interest,
            [],
        )
        all_handles.extend(waymo_utils.TRAJECTORY_HANDLES)
        plt.legend(handles=all_handles)

        def update(i):
            drawn_pts = []
            for (xs, ys), point in zip(data, points):
                if i < len(xs) and xs[i] is not None and ys[i] is not None:
                    point.set_data(xs[i], ys[i])
                    drawn_pts.append(point)
            return drawn_pts

        # Set Animation
        anim = FuncAnimation(
            fig, update, frames=range(1, max_len), blit=True, interval=100
        )
        out_path = os.path.join(
            os.path.abspath(target_path), f"scenario-{scenario_id}.mp4"
        )
        anim.save(out_path, writer=FFMpegWriter(fps=15))

    else:
        plt.legend(handles=all_handles)
        out_path = os.path.join(
            os.path.abspath(target_path), f"scenario-{scenario_id}.png"
        )
        fig = plt.gcf()
        fig.set_size_inches(1000 / 100, 1000 / 100)
        fig.savefig(out_path, dpi=100)

    print(f"Saving {out_path}")
    plt.close("all")
    return True


def dump_plots(target_base_path: str, scenario_dict, animate=False, filter_tags=None):
    """Plot and dump the map of multiple scenarios together at the target_path optionally filtered with tags.
    If animate is true, .mp4 video player files showing the animation of trajectories of track objects will be saved
    """
    try:
        os.makedirs(os.path.abspath(target_base_path))
        print(f"Created directory {target_base_path}")
    except FileExistsError:
        pass
    except (OSError, RuntimeError):
        print(f"{target_base_path} is an invalid path. Please enter a valid path")
        return

    plot_parameters = product(
        [(i + 1, scenario_id) for i, scenario_id in enumerate(scenario_dict)],
        [target_base_path],
        [scenario_dict],
        [animate],
        [filter_tags],
    )
    with Pool(min(cpu_count(), len(scenario_dict))) as pool:
        plots_dumped = pool.map(lambda x: save_plot(*x), list(plot_parameters))

    if any(plots_dumped):
        print(f"All images or recordings saved at {target_base_path}")
    else:
        print(f"No images or recordings saved as no tags matched")


def filter_scenario(
    scenario_tags: List[str],
    imported_tags: List[str],
    filter_tags: Tuple[List[str], int],
) -> bool:
    """Check if all tags in filter_tags exist in scenario_tags or imported_tags"""
    tags, filter_display = filter_tags
    if filter_display == 1:
        return all(x in scenario_tags for x in tags)
    elif filter_display == 2:
        return all(x in imported_tags for x in tags)
    return all(x in imported_tags + scenario_tags for x in tags)


def prompt_tags() -> Tuple[Optional[List[str]], bool]:
    """Prompt users to input the tags they want to add to the given scenarios"""
    while True:
        try:
            response = input("\nResponse: ")
            stripped_response = response.strip()
        except EOFError:
            print("Raised EOF. Attempting to exit browser.")
            return None, True
        if stripped_response == "":
            print(
                "Invalid response. Your response should have tags that are alphanumerical separated by Comma and can have special characters."
            )
        else:
            break

    tags = [tag.strip() for tag in stripped_response.lower().split(",")]

    return tags, False


def prompt_filter_tags() -> Tuple[Optional[List[str]], Optional[int], bool]:
    """Prompt users to enter tags they want to filter their response with"""
    filter_response = None
    print(
        f"\nDo you want to filter the output with scenario tags?:\n"
        "1. Yes, based on Tags Added\n"
        f"2. Yes, based on Imported Tags.\n"
        f"3. Yes, based on both tags merged.\n"
        f"4. No.\n"
        f"5. Go back to the Browser.\n"
        "Choose your response by entering 1, 2, 3, 4, 5.\n"
    )
    while filter_response is None:
        try:
            response = input("\nResponse: ")
            stripped_response = response.strip()
        except EOFError:
            print("Raised EOF. Attempting to exit browser.")
            break
        if re.compile("^[1-5]$", re.IGNORECASE).match(stripped_response):
            filter_response = int(stripped_response)
        else:
            print(
                "Invalid Response. Please choose your response by entering 1, 2, 3 or 4.\n"
            )
    if filter_response is None:
        return None, None, True

    if filter_response < 4:
        print(
            "What Tags do you want to filter the response with?\n"
            "Your response should have tags that are alphanumerical separated by Comma and can have special characters.\n"
        )
        tags, stop_browser = prompt_tags()
        if stop_browser:
            return None, None, True
    else:
        tags = None
    return tags, filter_response, False


def prompt_target_path(
    default_target_path: Optional[str] = None,
) -> Tuple[Optional[str], bool]:
    """Prompt users to enter the target path to which they want to save command output"""
    target_base_path = None
    valid_path = False
    if default_target_path is not None:
        print(
            f"Which path do you want to save the command output to?:\n"
            "1. Default Target Path.\n"
            f"2. Custom Target Path.\n"
            "3. Go back to the Browser.\n"
            "Choose your response by entering 1 ,2 or 3.\n"
        )
        valid_response = False
        while not valid_response:
            try:
                response = input("\nResponse: ")
                stripped_response = response.strip()
            except EOFError:
                print("Raised EOF. Attempting to exit browser.")
                return None, True
            if re.compile("^[1-3]$", re.IGNORECASE).match(stripped_response):
                if int(stripped_response) == 1:
                    target_base_path = default_target_path
                    valid_path = True
                if int(stripped_response) == 3:
                    return None, False
                valid_response = True
            else:
                print(
                    "Invalid Response. Please choose your response by entering 1 or 2.\n"
                )

    while not valid_path:
        try:
            response = input("\nEnter Path: ")
            stripped_path = response.strip("[ \"']")
        except EOFError:
            print("Raised EOF. Attempting to exit browser.")
            return None, True

        # Check if directory path is valid
        if not check_path_validity(stripped_path):
            print("Please enter a valid directory path\n")
            continue
        target_base_path = stripped_path
        valid_path = True
    return target_base_path, False


def prompt_export_before_exiting(
    tfrecords: List[Tuple[int, str]],
    tags_per_tfrecords: Dict[str, Dict[str, List[str]]],
    imported_tags: Dict[str, Dict[str, List[str]]],
) -> bool:
    """Prompt users whether they want to save and export the tags before exiting the browser"""
    filter_response = None
    print(
        f"\nDo you want to export the tags before exiting the browser?:\n"
        "1. Yes, for all TfRecords.\n"
        f"2. Yes, for select few.\n"
        f"3. No.\n"
        "Choose your response by entering 1, 2 or 3.\n"
    )
    while filter_response is None:
        try:
            response = input("\nResponse: ")
            stripped_response = response.strip()
        except EOFError:
            print(
                "Raised EOF. Your tags won't be saved unless explicitly exported before. Attempting to exit browser."
            )
            return True
        if re.compile("^[1-3]$", re.IGNORECASE).match(stripped_response):
            filter_response = int(stripped_response)
        else:
            print(
                "Invalid Response. Please choose your response by entering 1, 2, or 3.\n"
            )

    if filter_response == 3:
        print(
            "Any tags added in this session won't be saved, unless you have explicitly exported them before.\n"
        )
        return True
    else:
        tfr_paths = None
        if filter_response == 1:
            tfr_paths = [tfrecords[i][1] for i in range(len(tfrecords))]
        else:
            display_tf_records(tfrecords)
            print(
                "Enter the indexes of the tfrecords whose tags you want to export.\n"
                f"The indexes should be integers between between 1 and {len(tfrecords)} and should be separated by space.\n"
            )
            while True:
                try:
                    response = input("\nResponse: ")
                    stripped_response = response.strip()
                except EOFError:
                    print(
                        "Raised EOF. Your tags won't be saved unless explicitly exported before. Attempting to exit browser.\n"
                    )
                    return True
                if re.compile("^(\s*(\d+))+$", re.IGNORECASE).match(stripped_response):
                    input_lst = stripped_response.split()
                    valid_indexes = check_index_validity(
                        input_lst, len(tfrecords), "export"
                    )
                    if len(valid_indexes) == 0:
                        print(
                            f"Please enter valid indexes between 1 and {len(tfrecords)}.\n"
                        )
                        continue
                    tfr_paths = [tfrecords[i - 1][1] for i in valid_indexes]
                    break
                else:
                    print(
                        f"Invalid Response. Please enter indexes that should be integers between between 1 and {len(tfrecords)} and should be separated by space.\n"
                    )

        return export_tags_to_path(tfr_paths, tags_per_tfrecords, imported_tags)


def export_tags_to_path(
    tf_records: List[str],
    tags_per_tfrecords: Dict[str, Dict[str, List[str]]],
    imported_tags: Dict[str, Dict[str, List[str]]],
) -> bool:
    """Prompt users to enter the target path to .json file to which they want to export their tags"""
    tags_to_dump = {}
    for tfr_path in tf_records:
        tags_per_tfrecords.get(os.path.basename(tfr_path), {}),
        imported_tags.get(os.path.basename(tfr_path), {})

        if len(tags_per_tfrecords) == 0 and len(imported_tags) == 0:
            print(
                f"No tags for {os.path.basename(tfr_path)}. This TfRecord will be skipped."
            )
            continue

        print(
            f"Which tags do you want to export from {os.path.basename(tfr_path)}?:\n"
            "1. `Imported Tags` --> Tags imported from .json files.\n"
            f"2. `Tags Added` --> Tags added by you.\n"
            f"3. `Both Merged Together` --> Tags added by you and tags imported merged together.\n"
            "Choose your response by entering 1, 2 or 3.\n"
        )
        while True:
            try:
                response = input("\nResponse: ")
                stripped_response = response.strip()
            except EOFError:
                print(
                    "Raised EOF. Your tags won't be saved unless explicitly exported before. Attempting to exit browser."
                )
                return True
            if re.compile("^[1-3]$", re.IGNORECASE).match(stripped_response):
                if stripped_response == "1":
                    tags_to_dump.update(
                        {
                            os.path.basename(tfr_path): copy.deepcopy(
                                imported_tags.get(os.path.basename(tfr_path), {})
                            )
                        }
                    )
                elif stripped_response == "2":
                    tags_to_dump.update(
                        {
                            os.path.basename(tfr_path): copy.deepcopy(
                                tags_per_tfrecords.get(os.path.basename(tfr_path), {})
                            )
                        }
                    )
                else:
                    tags_to_dump.update(
                        {
                            os.path.basename(tfr_path): copy.deepcopy(
                                tags_per_tfrecords.get(os.path.basename(tfr_path), {})
                            )
                        }
                    )
                    scenario_imported_tags = {
                        os.path.basename(tfr_path): copy.deepcopy(
                            imported_tags.get(os.path.basename(tfr_path), {})
                        )
                    }
                    merge_tags(scenario_imported_tags, tags_to_dump)
                break
            else:
                print(
                    "Invalid Response. Please choose your response by entering 1, 2, or 3.\n"
                )

    if len(tags_to_dump) == 0:
        print("No tags available in any of the tfRecords to export.")
        return False

    print(
        "Enter the path to .json file to which you want to export the tags to?. If the file already exists, its data will be overwritten.:\n"
    )
    while True:
        try:
            response = input("\nEnter Path: ")
            stripped_path = response.strip("[ \"']")
        except EOFError:
            print(
                "Raised EOF. Your tags won't be saved unless explicitly exported before. Attempting to exit browser."
            )
            return True

        # Check if .json file path is valid
        if not check_path_validity(stripped_path) or not stripped_path.endswith(
            ".json"
        ):
            print("Please enter a valid .json file path\n")
            continue

        try:
            with open(os.path.abspath(stripped_path), "w") as f:
                json.dump(tags_to_dump, f, ensure_ascii=False, indent=4)
                print(f"All tags saved at {stripped_path}")
                break
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
    return False


def import_tags_from_path(
    imported_tags: Dict[str, Dict[str, List[str]]], json_filepath: str
):
    """
    Import the tags from json_filepath to this session
    """
    try:
        with open(json_filepath, "r") as f:
            new_tags = json.load(f)

    except (
        FileNotFoundError,
        IOError,
        OSError,
        json.decoder.JSONDecodeError,
    ):
        print(
            f"{json_filepath} does not exist or doesnt have the right permissions to read.\n"
            f"No tags will be imported from this filepath.\n"
        )
        return
    if len(new_tags) == 0:
        print(
            f"No data found in {json_filepath}. No tags will be imported from this filepath.\n"
        )
        return
    merge_tags(new_tags, imported_tags, True)


def display_scenario_tags(
    tags_per_scenarios: Dict[str, List[str]], tags_imported: Dict[str, List[str]]
):
    """
    Display the scenario tags and imported tags of this scenario
    """
    tag_data = []
    print("--------------------------------\n")
    for scenario_id in tags_per_scenarios:
        tag_data.append(
            [
                scenario_id,
                tags_per_scenarios.get(scenario_id, []),
                tags_imported.get(scenario_id, []),
            ]
        )
    print(
        tabulate(
            tag_data,
            headers=["Scenario ID", "Tags Added", "Tags Imported"],
        )
    )
    print("\n")


def merge_tags(new_imports: Dict, main_dict: Dict, display: bool = False):
    """Merge the tags in new_imports to main_dict and optionally display the new_import tags"""
    for tf_file in new_imports:
        if tf_file in main_dict:
            for scenario_id in new_imports[tf_file]:
                if scenario_id in main_dict[tf_file]:
                    main_dict[tf_file][scenario_id].extend(
                        [
                            tag.lower()
                            for tag in new_imports[tf_file][scenario_id]
                            if tag.lower() not in main_dict[tf_file][scenario_id]
                        ]
                    )
                else:
                    main_dict[tf_file][scenario_id] = new_imports[tf_file][scenario_id]
        else:
            main_dict[tf_file] = new_imports[tf_file]

        if display:
            print("\n-----------------------------------------------")
            print(f"Scenario Tags imported for {tf_file}:\n")
            display_scenario_tags(new_imports[tf_file], {})


def remove_tags(
    scenario_id: str,
    tags_to_remove: List[str],
    scenario_tags: List[str],
    imported: bool,
    remove_all: bool,
) -> List[str]:
    """Remove tags_to_remove from scenario_tags"""
    if imported:
        tags_list_name = "Imported Tags"
    else:
        tags_list_name = "Tags Added"

    if remove_all:
        print(f"All Tags removed from `{tags_list_name}` list of {scenario_id}")
        return []

    if len(scenario_tags) == 0:
        print(
            f"No tags added for {scenario_id} in `{tags_list_name}` list that can be removed"
        )
    else:
        old_len = len(scenario_tags)
        scenario_tags = [tag for tag in scenario_tags if tag not in tags_to_remove]
        if len(scenario_tags) == old_len:
            print(
                f"{scenario_id} doesn't have {tags_to_remove} in its `{tags_list_name}` list"
            )
        else:
            print(f"Tags removed from `{tags_list_name}` list of {scenario_id}")
    return scenario_tags


def get_scenario_and_tag_dict(
    tfrecord_file: str,
) -> Tuple[Dict[str, List[Union[scenario_pb2.Scenario, None]]], Dict[str, List]]:
    """Get scenario and tag dictionary having info and tags for all scenarios in this tfrecord_file"""
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


def parse_tfrecords(
    tfrecord_paths: List[str],
) -> Tuple[Dict[str, None], Dict[str, None]]:
    """Parse every tfrecord paths passed by users and create hash tables of scenarios and tags for every tfrecord file found"""
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


def display_tf_records(records: List[Tuple[int, str]]):
    """Display the tf_records that were loaded in a tabular format"""
    print("\nWaymo tfRecords:\n")
    print(
        tabulate(
            records,
            headers=["Index", "TfRecords"],
        )
    )
    print("\n\n")


def display_scenarios_in_tfrecord(
    tfrecord_path: str,
    scenario_dict: Dict,
    tfrecord_tags: Dict[str, List[str]],
    tags_imported: Dict[str, List[str]],
    filter_tags: Optional[Tuple[List[str], int]] = None,
) -> List[str]:
    """Display all the scenarios of a tf_record file and their info in tabular format"""
    scenario_data_lst = []
    scenario_counter = 1
    scenario_ids = []
    for scenario_id in scenario_dict:
        if filter_tags:
            if not filter_scenario(
                tfrecord_tags.get(scenario_id, []),
                tags_imported.get(scenario_id, []),
                filter_tags,
            ):
                scenario_counter += 1
                continue
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
    if len(scenario_data_lst):
        print("\n-----------------------------------------------")
        print(f"{len(scenario_dict)} scenarios in {os.path.basename(tfrecord_path)}:\n")
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
    if len(scenario_data_lst) == 0:
        if filter_tags is not None:
            print(f"No scenarios in {tfrecord_path} have the tags {filter_tags[0]}")
        else:
            print(f"No scenarios found in {tfrecord_path}")
    return scenario_ids


def export_scenario(
    target_base_path: str, tfrecord_file_path: str, scenario_id: str
) -> bool:
    """Export the scenario.py and waymo.yaml file of the scenario with scenario_id to target_base_path/<scenario_id> folder"""
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
        with open(scenario_py, "w") as f:
            f.write(
                waymo_utils.gen_smarts_scenario_code(tfrecord_file_path, scenario_id)
            )
        print(f"Scenario.py created in {subfolder_path}.")
    return True


def check_index_validity(
    input_arg: List[str], upper_limit: int, command_type: str
) -> List[int]:
    """Check if input_arg passed by user is and integer and within the upper_limit"""
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
    """Check if target base path is valid"""
    try:
        Path(target_base_path).resolve()
    except (IOError, OSError, RuntimeError):
        print(
            f"{target_base_path} is an invalid path. Please enter a valid directory path"
        )
        return False
    return True


def tfrecords_browser(
    tfrecord_paths: List[str],
    default_target_path: Optional[str] = None,
    tags_json: Optional[str] = None,
) -> None:
    """TfRecord Browser, which shows all the tfrecord files loaded in and takes
    in commands that can be run by the user at this level
    """
    scenarios_per_tfrecords, tags_per_tfrecords = parse_tfrecords(tfrecord_paths)
    imported_tags = {}
    if tags_json:
        import_tags_from_path(imported_tags, tags_json)
    if not scenarios_per_tfrecords:
        print("No .tfrecord files exist in paths provided. Please pass valid paths.")
        return

    tf_records = []
    tf_counter = 1
    for tf in scenarios_per_tfrecords:
        tf_records.append((tf_counter, tf))
        tf_counter += 1
    stop_browser = False
    print_commands = True
    while not stop_browser:
        if print_commands:
            display_tf_records(tf_records)
            print(
                "TfRecords Browser.\n"
                "You can use the following commands to further explore these datasets:\n"
                "1. `display all` --> Displays the info of all the scenarios from every tfRecord file together\n"
                "                     Displays can be filtered on the basis of tags in a subsequent option.\n"
                f"2. `display <indexes>` --> Displays the info of tfRecord files at these indexes of the table.\n"
                f"                           The indexes should be an integer between 1 and {len(tf_records)} and space separated.\n"
                "                            Displays can be filtered on the basis of tags.\n"
                f"3. `explore <index>` --> Explore the tfRecord file at this index of the table.\n"
                f"                         The index should be an integer between 1 and {len(tf_records)}\n"
                f"4. `import tags` --> Import the tags of tfRecords from a previously saved .json file.\n"
                f"                     Only tags of tfRecords which are displayed above will be imported. Ensure the name of tfRecord match with the ones displayed above.\n"
                f"5. `export tags all/<indexes>` --> Export the tags of the tfRecords at these indexes to a .json file.\n"
                f"                                   Optionally you can use all instead to export tags of all tfRecords. The path to the .json file should be valid.\n"
                "6. `exit` --> Exit the program\n"
            )
        print_commands = True
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

            # Prompt user to filter response by tags
            tags, filter_display, stop_browser = prompt_filter_tags()
            if stop_browser or filter_display == 5:
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
                    (tags, filter_display) if tags is not None else None,
                )
                print("\n")

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
                tfr_paths = [tf_records[i][1] for i in range(len(tf_records))]
            else:
                valid_indexes = check_index_validity(
                    input_lst[2:], len(tf_records), "export"
                )
                if len(valid_indexes) == 0:
                    continue
                tfr_paths = [tf_records[i - 1][1] for i in valid_indexes]

            stop_browser = export_tags_to_path(
                tfr_paths, tags_per_tfrecords, imported_tags
            )

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

            imported_tags[os.path.basename(tfr_path)] = imported_tags.get(
                os.path.basename(tfr_path), {}
            )
            stop_browser = explore_tf_record(
                tfr_path,
                scenarios_per_tfrecords[tfr_path],
                tags_per_tfrecords[os.path.basename(tfr_path)],
                imported_tags[os.path.basename(tfr_path)],
                default_target_path,
            )

        elif user_input.lower() == "exit":
            stop_browser = True

        else:
            print(
                "Invalid command. Please enter a valid command. See command formats above"
            )
            print_commands = False
    if len(tags_per_tfrecords) or len(imported_tags):
        prompt_export_before_exiting(tf_records, tags_per_tfrecords, imported_tags)
    print(
        "If you exported any scenarios, you can build them using the command `scl scenario build <target_base_path>`.\n"
        "Have a look at README.md at the root level of this repo for more info on how to build scenarios."
    )
    print("Exiting the Browser")


def explore_tf_record(
    tfrecord: str,
    scenario_dict: Dict,
    tfrecord_tags: Dict[str, List[str]],
    imported_tfrecord_tags: Dict[str, List[str]],
    default_target_path: Optional[str] = None,
) -> bool:
    """Tf Record Explorer, which shows all the scenarios and their info and
    takes in command that can be run by the user at this level"""
    print("TfRecord Explorer")
    stop_exploring = False
    print_commands = True
    scenario_ids = None
    while not stop_exploring:
        if print_commands:
            scenario_ids = display_scenarios_in_tfrecord(
                tfrecord,
                scenario_dict,
                tfrecord_tags,
                imported_tfrecord_tags,
            )
            print("\n")
            print(
                f"{os.path.basename(tfrecord)} TfRecord Browser.\n"
                f"You can use the following commands to further explore these scenarios:\n"
                "1. `display` --> Display the scenarios in this tfrecord filtered based on the tags chosen in a subsequent option.\n"
                "2. `explore <index>` --> Select and explore further the scenario at this index of the table.\n"
                f"                        The index should be an integer between 1 and {len(scenario_ids)}\n"
                "3. `export all/<indexes>` --> Export the scenarios at these indexes or all of the table to a target path\n"
                f"                             The indexes should be an integer between 1 and {len(scenario_ids)} separated by space\n"
                f"                             The exports can be filtered based on the tags chosen in a subsequent option.\n"
                "4. `preview all` --> Plot and dump the images of the map of all scenarios in this tf_record to a target path.\n"
                "5. `preview` or `preview <indexes>` --> Plot and display the maps of these scenarios at these indexes of the table  (or all the scenarios if just `preview`) .\n"
                f"                                       The indexes should be an integer between 1 and {len(scenario_ids)} and should be separated by space.\n"
                f"6. `animate all` --> Plot and dump the animations the trajectories of objects on map of all scenarios in this tf_record to a target path.\n"
                f"7. `animate` or `animate <indexes>` --> Plot the map and animate the trajectories of objects of all scenarios if just `animate` or scenario at these indexes of the table.\n"
                f"                                        The indexes should be an integer between 1 and {len(scenario_ids)} and should be separated by space.\n"
                f"8. `tag all/<indexes>` or `tag imported all/<indexes>` --> Tag the scenarios at these indexes of the table or all with tags mentioned.\n"
                f"                                                           Optionally if you call with `tag imported` then the tags for these scenarios will be added to imported tag list.\n"
                f"                                                           If indexes, then they need to be integers between 1 and {len(scenario_ids)} and should be separated by space.\n"
                f"9. `untag all/<indexes>` or `untag imported all/<indexes>` --> Untag the scenarios at theses indexes of the table or all with tags mentioned.\n"
                f"                                                               Optionally if you call with `untag imported` then the tags for these scenarios will be removed from imported tag list.\n"
                f"                                                               If indexes, then they need to be integers between 1 and {len(scenario_ids)} and should be separated by space.\n"
                "10. `back` --> Go back to the tfrecords browser\n"
                "11. `exit` --> Exit the program\n"
            )
        print_commands = True
        print("\n")
        try:
            raw_input = input("Command: ")
            user_input = raw_input.strip()
        except EOFError:
            print("Raised EOF. Attempting to exit browser.")
            return True

        if user_input.lower() == "display":
            display_scenario_tags(tfrecord_tags, imported_tfrecord_tags)
            tags, filter_display, stop_browser = prompt_filter_tags()
            if stop_browser or filter_display == 5:
                continue

        elif re.compile("^export[\s]+(all|(?:\s*(\d+))+)", flags=re.IGNORECASE).match(
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

            # Prompt to allow filtering response with tags
            display_scenario_tags(tfrecord_tags, imported_tfrecord_tags)
            tags, filter_export, stop_browser = prompt_filter_tags()
            if stop_browser:
                return True
            if filter_export == 5:
                continue

            print(
                "Enter the path to directory to which you want to export the scenarios:"
            )
            # Prompt to input target base path
            target_base_path, stop_browser = prompt_target_path(default_target_path)
            if stop_browser:
                return True
            if not target_base_path:
                continue

            # Try exporting the scenario
            exported = False
            for idx in valid_indexes:
                if tags is not None:
                    if not filter_scenario(
                        tfrecord_tags.get(scenario_ids[idx - 1], []),
                        imported_tfrecord_tags.get(scenario_ids[idx - 1], []),
                        (tags, filter_export),
                    ):
                        continue
                exported = (
                    export_scenario(target_base_path, tfrecord, scenario_ids[idx - 1])
                    or exported
                )
            if exported:
                print(
                    f"\nYou can build these scenarios exported using the command `scl scenario build-all {target_base_path}`"
                )
            else:
                if tags:
                    print("No scenarios were exported since no tags matched\n")

        elif re.compile("^preview[\s]+all$", flags=re.IGNORECASE).match(user_input):
            display_scenario_tags(tfrecord_tags, imported_tfrecord_tags)
            tags, filter_preview, stop_browser = prompt_filter_tags()
            if stop_browser:
                return True
            if filter_preview == 5:
                continue

            print(
                "Enter the path to directory to which you want to dump the images of the maps of scenarios?:"
            )
            target_base_path, stop_browser = prompt_target_path(default_target_path)
            if stop_browser:
                return True
            if not target_base_path:
                continue

            # Dump all the scenario plots of this tfrecord file to this target base path
            print(
                f"Plotting and dumping all the scenario maps in {os.path.basename(tfrecord)} tfrecord file"
            )
            dump_plots(
                target_base_path,
                scenario_dict,
                (tags, filter_preview, tfrecord_tags, imported_tfrecord_tags)
                if tags
                else None,
            )

        elif user_input.lower() == "preview" or re.compile(
            "^preview[\s]+(?:\s*(\d+))+$", flags=re.IGNORECASE
        ).match(user_input):
            input_lst = user_input.split()
            if len(input_lst) == 1:
                valid_indexes = [i for i in range(len(scenario_ids))]
            else:
                # Check if index passed is valid
                valid_indexes = check_index_validity(
                    input_lst[1:], len(scenario_ids), "preview"
                )
            if len(valid_indexes) == 0:
                continue

            display_scenario_tags(tfrecord_tags, imported_tfrecord_tags)
            tags, filter_preview, stop_browser = prompt_filter_tags()
            if stop_browser:
                return True

            if filter_preview == 5:
                continue

            # Plot the maps of these scenarios
            scenarios_to_plot = []
            for i in range(len(valid_indexes)):
                scenario_idx = scenario_ids[valid_indexes[i] - 1]
                if tags is not None:
                    if not filter_scenario(
                        tfrecord_tags.get(scenario_idx, []),
                        imported_tfrecord_tags.get(scenario_idx, []),
                        (tags, filter_preview),
                    ):
                        continue
                if scenario_dict[scenario_idx][1] is None:
                    scenario_dict[scenario_idx][1] = get_map_features_for_scenario(
                        scenario_dict[scenario_idx][0]
                    )
                scenarios_to_plot.append(
                    (scenario_dict[scenario_idx], valid_indexes[i])
                )

            if len(scenarios_to_plot) > 0:
                plot_parameters = product(
                    scenarios_to_plot,
                    [False],
                    [None],
                )
                with Pool(min(cpu_count(), len(scenarios_to_plot))) as pool:
                    pool.map(lambda x: plot_scenario(*x), list(plot_parameters))

            else:
                print("No map images were plotted as no filter tags matched\n")

        elif re.compile("^animate[\s]+all$", flags=re.IGNORECASE).match(user_input):
            display_scenario_tags(tfrecord_tags, imported_tfrecord_tags)
            tags, filter_animate, stop_browser = prompt_filter_tags()
            if stop_browser:
                return True
            if filter_animate == 5:
                continue

            print(
                "Enter the path to directory to which you want to dump the animations of the track objects of scenarios?:"
            )
            target_base_path, stop_browser = prompt_target_path(default_target_path)
            if stop_browser:
                return True
            if not target_base_path:
                continue

            # Dump all the scenario plots of this tfrecord file to this target base path
            print(
                f"Plotting and dumping all the scenarios animations in {os.path.basename(tfrecord)} tfrecord file"
            )
            dump_plots(
                target_base_path,
                scenario_dict,
                True,
                (tags, filter_animate, tfrecord_tags, imported_tfrecord_tags)
                if tags
                else None,
            )

        elif user_input.lower() == "animate" or re.compile(
            "^animate[\s]+(?:\s*(\d+))+$", flags=re.IGNORECASE
        ).match(user_input):
            input_lst = user_input.split()
            if len(input_lst) == 1:
                valid_indexes = [i for i in range(len(scenario_ids))]
            else:
                # Check if index passed is valid
                valid_indexes = check_index_validity(
                    input_lst[1:], len(scenario_ids), "animate"
                )
            if len(valid_indexes) == 0:
                continue

            display_scenario_tags(tfrecord_tags, imported_tfrecord_tags)
            tags, filter_animate, stop_browser = prompt_filter_tags()
            if stop_browser:
                return True
            if filter_animate == 5:
                continue

            # Animate the maps of these scenarios
            scenarios_to_animate = []
            for i in range(len(valid_indexes)):
                scenario_idx = scenario_ids[valid_indexes[i] - 1]
                if tags is not None:
                    if not filter_scenario(
                        tfrecord_tags.get(scenario_idx, []),
                        imported_tfrecord_tags.get(scenario_idx, []),
                        (tags, filter_animate),
                    ):
                        continue
                if scenario_dict[scenario_idx][1] is None:
                    scenario_dict[scenario_idx][1] = get_map_features_for_scenario(
                        scenario_dict[scenario_idx][0]
                    )

                if scenario_dict[scenario_idx][2] is None:
                    scenario_dict[scenario_idx][2] = get_trajectory_data(
                        scenario_dict[scenario_idx][0]
                    )
                scenarios_to_animate.append(
                    (scenario_dict[scenario_idx], valid_indexes[i])
                )

            if len(scenarios_to_animate) > 0:
                plot_parameters = product(
                    scenarios_to_animate,
                    [True],
                    [None],
                )
                with Pool(min(cpu_count(), len(scenarios_to_animate))) as pool:
                    pool.map(lambda x: plot_scenario(*x), list(plot_parameters))

            else:
                print("No animations were shown as no filter tags matched\n")

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
            tags, stop_browser = prompt_tags()
            if stop_browser:
                return True

            if imported:
                for i in range(len(valid_indexes)):
                    scenario_idx = scenario_ids[valid_indexes[i] - 1]
                    if scenario_idx in imported_tfrecord_tags:
                        imported_tfrecord_tags[
                            scenario_idx
                        ].extend(  # pytype: disable=attribute-error
                            [
                                tag
                                for tag in tags
                                if tag not in imported_tfrecord_tags[scenario_idx]
                            ]
                        )
                    else:
                        if tags is not None:
                            imported_tfrecord_tags[scenario_idx] = tags
                        else:
                            imported_tfrecord_tags[scenario_idx] = []
                print("Tags added to `Imported Tags` list")

            else:
                for i in range(len(valid_indexes)):
                    scenario_idx = scenario_ids[valid_indexes[i] - 1]
                    tfrecord_tags[scenario_idx].extend(
                        [tag for tag in tags if tag not in tfrecord_tags[scenario_idx]]
                    )
                print("Tags added to `Tags Added` list")

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
                "Optionally you can respond `remove all`, to remove all tags from these scenarios.\n"
            )
            tags, stop_browser = prompt_tags()
            if stop_browser:
                return True

            # if response is remove all, remove all the tags from the scenarios mentioned
            remove_all = False
            if "remove all" in tags:
                remove_all = True

            if imported:
                for i in range(len(valid_indexes)):
                    scenario_idx = scenario_ids[valid_indexes[i] - 1]
                    if scenario_idx in imported_tfrecord_tags:
                        imported_tfrecord_tags[scenario_idx] = remove_tags(
                            scenario_idx,
                            tags,
                            imported_tfrecord_tags[scenario_idx],
                            True,
                            remove_all,
                        )
                    else:
                        print(f"No imported tags for {scenario_idx}")

            else:
                for i in range(len(valid_indexes)):
                    scenario_idx = scenario_ids[valid_indexes[i] - 1]
                    if len(tfrecord_tags[scenario_idx]) == 0:
                        print(f"No tags added for {scenario_idx} that can be removed")
                    else:
                        tfrecord_tags = remove_tags(
                            scenario_idx,
                            tags,
                            tfrecord_tags[scenario_idx],
                            False,
                            remove_all,
                        )

        elif re.compile("^explore[\s]+[\d]+$", flags=re.IGNORECASE).match(user_input):
            input_lst = user_input.split()

            # Check if index passed is valid
            valid_indexes = check_index_validity(
                input_lst[1:], len(scenario_ids), "explore"
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
            imported_tfrecord_tags[scenario_id] = imported_tfrecord_tags.get(
                scenario_id, []
            )
            exit_browser = explore_scenario(
                tfrecord,
                scenario_ids.index(scenario_id),
                scenario_dict[scenario_id],
                tfrecord_tags[scenario_id],
                imported_tfrecord_tags[scenario_id],
                default_target_path,
            )
            if exit_browser:
                return True

        elif user_input.lower() == "back":
            stop_exploring = True
            print("Going back to the tfRecords browser")
            continue

        elif user_input.lower() == "exit":
            return True
        else:
            print(
                "Invalid command. Please enter a valid command. See command formats above"
            )
            print_commands = False

    return False


def explore_scenario(
    tfrecord_file_path: str,
    scenario_index: int,
    scenario_info: List,
    scenario_tags: List[str],
    imported_scenario_tags: List[str],
    default_target_path: Optional[str],
) -> bool:
    """Scenario Explorer, which shows map features and track objects info of the scenario
    and takes in commands that can be run by the user at this level
    """

    scenario = scenario_info[0]
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
    print("Scenario Explorer")

    def display_scenario_data_info():
        print("\n")
        print("-----------------------------------------------")
        print(f"Scenario {scenario.scenario_id} Map Features:\n")
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
        print("\nObject of Interest Ids: ", [i for i in scenario.objects_of_interest])

    display_scenario_data_info()

    stop_exploring = False
    print_commands = True
    while not stop_exploring:
        if print_commands:
            print(
                f"\n\nScenario {scenario.scenario_id}.\n"
                "You can use the following commands to further this scenario:\n"
                f"1. `display --> Display the scenario info which includes the map feature ids and track ids.\n"
                f"2. `export` --> Export the scenario to a target base path asked to input in a subsequent option.\n"
                f"3. `preview` or `preview <feature_ids>` --> Plot and display the map of the scenario with the feature ids highlighted in Blue if passed.\n"
                f"                                            The feature ids need to be separated by space, be numbers from the map feature ids mentioned above and will not be highlighted if they dont exist.\n"
                "4. `animate` or `animate <track_ids> --> Animate the trajectories of track objects on the map of this scenario with the track ids highlighted in Red if passed.\n"
                f"                                        The track ids need to be separated by space, be numbers from the track object ids mentioned above and will not be highlighted if they dont exist.\n"
                f"5. `tag` or `tag imported` --> Tag the scenario with tags mentioned.\n"
                "                                Optionally if you call with `tag imported` then the tags will be added to imported tag list.\n"
                f"6. `untag` or `untag imported` --> Untag the scenario with tags mentioned.\n"
                "                                    Optionally if you call with `untag imported` then the tags will be removed to imported tag list.\n"
                "7. `back` --> Go back to this scenario's tfrecord browser.\n"
                "8. `exit` --> Exit the program\n"
            )
            print_commands = False
        try:
            raw_input = input("\nCommand: ")
            user_input = raw_input.strip()
        except EOFError:
            print("Raised EOF. Attempting to exit browser.")
            return True

        if user_input.lower() == "display":
            display_scenario_data_info()
            print_commands = True

        elif user_input.lower() == "export":
            print(
                "Enter the path to directory to which you want to export this scenario:"
            )
            # Prompt users to input target base path
            target_base_path, stop_browser = prompt_target_path(default_target_path)
            if stop_browser:
                return True
            if not target_base_path:
                continue

            # Try exporting the scenario to the target_base_path
            export_scenario(target_base_path, tfrecord_file_path, scenario.scenario_id)
            print(
                f"\nYou can build the scenario exported using the command `scl scenario build {target_base_path}`"
            )
            print_commands = True

        elif user_input.lower() == "preview" or re.compile(
            "^preview[\s]+(?:\s*(\d+))+$", flags=re.IGNORECASE
        ).match(user_input):
            # Plot the map of this scenario
            input_lst = user_input.split()
            feature_ids = None
            if len(input_lst) > 1:
                feature_ids = input_lst[1:]
            plot_scenario((scenario_info, scenario_index), False, feature_ids)

        elif user_input.lower() == "animate" or re.compile(
            "^animate[\s]+(?:\s*(\d+))+?$", flags=re.IGNORECASE
        ).match(user_input):
            # Plot and animate this scenario
            input_lst = user_input.split()
            track_ids = None
            if len(input_lst) > 1:
                track_ids = input_lst[1:]
            plot_scenario((scenario_info, scenario_index), True, track_ids)

        elif re.compile("^tag([\s]+imported)?$", flags=re.IGNORECASE).match(user_input):
            input_lst = user_input.lower().split()
            imported = True if "imported" in input_lst else False

            print(
                "What Tags do you want to add?\n"
                "Your response should have tags that are alphanumerical and can have special characters but need to be separated by Comma."
            )
            tags, stop_browser = prompt_tags()
            if stop_browser:
                return True

            if imported:
                imported_scenario_tags.extend(
                    [tag for tag in tags if tag not in imported_scenario_tags]
                )
                print("Tags added to `Imported Tags` list")
            else:
                scenario_tags.extend([tag for tag in tags if tag not in scenario_tags])
                print("Tags added to `Tags Added` list")
            print_commands = True

        elif re.compile("^untag([\s]+imported)?$", flags=re.IGNORECASE).match(
            user_input
        ):
            input_lst = user_input.lower().split()
            imported = True if "imported" in input_lst else False

            print(
                "What Tags do you want to remove?. Tags that don't exist wont be removed.\n"
                "Your response should have tags that are alphanumerical and can have special characters but need to be separated by Comma."
                "Optionally you can respond `remove all`, to remove all tags from these scenarios.\n"
            )
            tags, stop_browser = prompt_tags()
            if stop_browser:
                return True

            remove_all = False
            if "remove all" in tags:
                remove_all = True

            scenario_idx = scenario.scenario_id
            if imported:
                imported_scenario_tags = remove_tags(
                    scenario_idx, tags, imported_scenario_tags, True, remove_all
                )
            else:
                scenario_tags = remove_tags(
                    scenario_idx, tags, scenario_tags, False, remove_all
                )
            print_commands = True

        elif user_input.lower() == "back":
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

    warnings.warn(
        "waymo_browser.py has been deprecated in favour of the scl waymo command line tools.",
        category=DeprecationWarning,
    )
    readline.set_completer_delims(" \t\n;")
    readline.parse_and_bind("tab: complete")
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    from types import ModuleType

    assert isinstance(
        scenario_pb2, ModuleType
    ), "Module not installed please see warnings."

    parser = argparse.ArgumentParser(
        prog="waymo_browser.py",
        description="Text based TfRecords Browser.",
    )
    parser.add_argument(
        "tfrecords",
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
    parser.add_argument(
        "--import-tags",
        help="Import tags for tfRecord scenarios from this .json file",
        type=str,
        default=None,
    )
    args = parser.parse_args()
    valid_tf_paths = []
    if args.target_base_path is not None:
        if not os.path.exists(os.path.abspath(args.target_base_path)):
            print(
                f"Default Target Base Path {args.target_base_path} does not exist.\n"
                f"Please make sure Default Target Base path passed is valid and it exists."
            )
            exit()
    if args.import_tags is not None:
        if not os.path.exists(
            os.path.abspath(args.import_tags)
        ) or not args.import_tags.endswith(".json"):
            print(
                f".json file  {args.import_tags} does not exist.\n"
                f"Please make sure .json file path passed is valid and it exists."
            )
            exit()
    for tf_path in args.tfrecords:
        if not os.path.exists(os.path.abspath(tf_path)):
            print(
                f"Path {tf_path} does not exist and hence wont be browsed.\n"
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
            os.path.abspath(args.import_tags) if args.import_tags else None,
        )
