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
# FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D

from smarts.core.utils.file import read_tfrecord_file


class WaymoDatasetError(Exception):
    """Represents an error related to the data in a Waymo dataset scenario."""

    pass


MAP_HANDLES = [
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

TRAJECTORY_HANDLES = [
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
        color="yellow",
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


def _create_interactive_handle(object_type: int, v_id: int) -> Line2D:
    if object_type == 1:
        return Line2D(
            [],
            [],
            color="green",
            marker="^",
            linestyle="None",
            markersize=5,
            label=f"Interactive Car {v_id}",
        )
    elif object_type == 2:
        return Line2D(
            [],
            [],
            color="green",
            marker="d",
            linestyle="None",
            markersize=5,
            label=f"Interactive Pedestrian {v_id}",
        )
    elif object_type == 3:
        return Line2D(
            [],
            [],
            color="green",
            marker="*",
            linestyle="None",
            markersize=5,
            label=f"Interactive Cyclist {v_id}",
        )
    else:
        Line2D(
            [],
            [],
            color="green",
            marker="8",
            linestyle="None",
            markersize=5,
            label=f"Interactive Other {v_id}",
        )


def _plot_map_features(map_features: Dict):
    for lane in map_features["lane"]:
        pts = np.array([[p.x, p.y] for p in lane[0].polyline])
        plt.plot(pts[:, 0], pts[:, 1], linestyle=":", color="gray")

    for road_line in map_features["road_line"]:
        pts = np.array([[p.x, p.y] for p in road_line[0].polyline])
        xs, ys = pts[:, 0], pts[:, 1]
        if road_line[0].type in [1, 4, 5]:
            plt.plot(pts[:, 0], pts[:, 1], "y--")
        else:
            plt.plot(pts[:, 0], pts[:, 1], "y-")

    for road_edge in map_features["road_edge"]:
        pts = np.array([[p.x, p.y] for p in road_edge[0].polyline])
        plt.plot(pts[:, 0], pts[:, 1], "k-")

    for crosswalk in map_features["crosswalk"]:
        poly_points = [[p.x, p.y] for p in crosswalk[0].polygon]
        poly_points.append(poly_points[0])
        pts = np.array(poly_points)
        plt.plot(pts[:, 0], pts[:, 1], "k--")

    for speed_bump in map_features["speed_bump"]:
        poly_points = [[p.x, p.y] for p in speed_bump[0].polygon]
        poly_points.append(poly_points[0])
        pts = np.array(poly_points)
        plt.plot(pts[:, 0], pts[:, 1], "k:")

    for stop_sign in map_features["stop_sign"]:
        plt.scatter(
            stop_sign[0].position.x,
            stop_sign[0].position.y,
            marker="o",
            c="red",
            alpha=1,
        )


def _get_map_features(scenario) -> Dict[str, List]:
    map_features = defaultdict(lambda: [])
    for i in range(len(scenario.map_features)):
        map_feature = scenario.map_features[i]
        key = map_feature.WhichOneof("feature_data")
        if key is not None:
            map_features[key].append((getattr(map_feature, key), str(map_feature.id)))
    return map_features


def _get_trajectories(scenario) -> Dict[int, Dict[str, Any]]:
    num_steps = len(scenario.timestamps_seconds)
    trajectories = defaultdict(lambda: {"positions": [None] * num_steps})
    for i in range(len(scenario.tracks)):
        vehicle_id = scenario.tracks[i].id
        trajectories[vehicle_id]["is_ego"] = i == scenario.sdc_track_index
        trajectories[vehicle_id]["object_type"] = scenario.tracks[i].object_type
        for j in range(num_steps):
            obj_state = scenario.tracks[i].states[j]
            trajectories[vehicle_id]["positions"][j] = (
                obj_state.center_x if obj_state.valid else None,
                obj_state.center_y if obj_state.valid else None,
            )
    return trajectories


def _plot_trajectories(
    trajectories: Dict[int, Dict[str, Any]],
    interactive_ids: List[int],
) -> Tuple[List[Line2D], List[Optional[Tuple[list, list]]], List[Line2D]]:
    points, data, handles = [], [], []

    # Need to plot something initially to get handles to the point objects,
    # so just use a valid point from the first trajectory
    first_traj = list(trajectories.values())[0]["positions"]
    ind = None
    for i in range(len(first_traj)):
        if first_traj[i][0] is not None:
            ind = i
            break
    assert ind is not None, "No valid point in first trajectory"
    x0 = first_traj[ind][0]
    y0 = first_traj[ind][1]
    for v_id, props in trajectories.items():
        xs = [p[0] for p in props["positions"]]
        ys = [p[1] for p in props["positions"]]

        if props["is_ego"]:
            (point,) = plt.plot(x0, y0, "c^")
            continue

        is_interactive = int(v_id) in interactive_ids
        object_type = props["object_type"]
        if is_interactive:
            handles.append(_create_interactive_handle(object_type, v_id))

        if object_type == 1:
            if is_interactive:
                (point,) = plt.plot(x0, y0, "g^")
            else:
                (point,) = plt.plot(x0, y0, "k^")
        elif object_type == 2:
            if is_interactive:
                (point,) = plt.plot(x0, y0, "gd")
            else:
                (point,) = plt.plot(x0, y0, "md")
        elif object_type == 3:
            if is_interactive:
                (point,) = plt.plot(x0, y0, "g*")
            else:
                (point,) = plt.plot(x0, y0, "y*")
        else:
            if is_interactive:
                (point,) = plt.plot(x0, y0, "g8")
            else:
                (point,) = plt.plot(x0, y0, "k8")
        data.append((xs, ys))
        points.append(point)
    return points, data, handles


def get_tfrecord_info(tfrecord_file: str) -> Dict[str, Dict[str, Any]]:
    """Extract info about each scenario in the TFRecord file."""
    from waymo_open_dataset.protos import scenario_pb2

    scenarios = dict()
    records = read_tfrecord_file(tfrecord_file)
    for record in records:
        scenario = scenario_pb2.Scenario()
        scenario.ParseFromString(bytearray(record))
        scenario_id = scenario.scenario_id
        num_vehicles = 0
        num_pedestrians = 0
        for track in scenario.tracks:
            if track.object_type == 1:
                num_vehicles += 1
            elif track.object_type == 2:
                num_pedestrians += 1

        scenarios[scenario_id] = {
            "timestamps": len(scenario.timestamps_seconds),
            "vehicles": num_vehicles,
            "pedestrians": num_pedestrians,
        }
    return scenarios


def plot_scenario(
    tfrecord_file: str,
    scenario_id: str,
    animate: bool,
    label_vehicles: bool,
):
    """Plot the map features of a Waymo scenario,
    and optionally plot/animate the vehicle trajectories."""
    from smarts.core.waymo_map import WaymoMap

    source = f"{tfrecord_file}#{scenario_id}"
    scenario = WaymoMap.parse_source_to_scenario(source)

    fig = plt.figure()
    mng = plt.get_current_fig_manager()
    mng.resize(1000, 1000)
    map_features = _get_map_features(scenario)
    handles = MAP_HANDLES
    if label_vehicles:
        handles.extend(TRAJECTORY_HANDLES)
        trajectories = _get_trajectories(scenario)
        for v_id, props in trajectories.items():
            valid_pts = [p for p in props["positions"] if p[0] is not None]
            if len(valid_pts) > 0:
                x = valid_pts[0][0]
                y = valid_pts[0][1]
                plt.scatter(x, y, marker="o", c="blue")
                bbox_props = dict(boxstyle="square,pad=0.1", fc="white", ec=None)
                plt.text(x + 1, y + 1, f"{v_id}", bbox=bbox_props)
    elif animate:
        trajectories = _get_trajectories(scenario)
        interactive_ids = [i for i in scenario.objects_of_interest]
        points, data, interactive_handles = _plot_trajectories(
            trajectories, interactive_ids
        )
        handles.extend(TRAJECTORY_HANDLES + interactive_handles)

        def update(i):
            drawn_pts = []
            for (xs, ys), point in zip(data, points):
                if i < len(xs) and xs[i] is not None and ys[i] is not None:
                    point.set_data(xs[i], ys[i])
                    drawn_pts.append(point)
            return drawn_pts

        num_steps = len(scenario.timestamps_seconds)
        anim = FuncAnimation(
            fig, update, frames=range(1, num_steps), blit=True, interval=100
        )
    _plot_map_features(map_features)
    plt.title(f"Scenario {scenario_id}")
    plt.legend(handles=handles)
    plt.axis("equal")
    plt.show()


def gen_smarts_scenario_code(dataset_path: str, scenario_id: str) -> str:
    """Generate source code for the scenario.py of a SMARTS scenario for a Waymo scenario."""
    return f"""from pathlib import Path
from smarts.sstudio import gen_scenario
from smarts.sstudio import types as t

dataset_path = "{dataset_path}"
scenario_id = "{scenario_id}"
traffic_histories = [
    t.TrafficHistoryDataset(
        name=f"waymo",
        source_type="Waymo",
        input_path=dataset_path,
        scenario_id=scenario_id,
    )
]

gen_scenario(
    t.Scenario(
        map_spec=t.MapSpec(
            source=f"{{dataset_path}}#{{scenario_id}}", lanepoint_spacing=1.0
        ),
        traffic_histories=traffic_histories,
    ),
    output_dir=Path(__file__).parent,
)
"""
