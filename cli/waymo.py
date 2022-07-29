from collections import defaultdict
import os
from typing import Any, Dict, List, Optional, Tuple

import click
from matplotlib.animation import FuncAnimation
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from smarts.core.utils.file import read_tfrecord_file
from smarts.core.waymo_map import WaymoMap
from smarts.sstudio.types import MapSpec
from tabulate import tabulate
from waymo_open_dataset.protos import scenario_pb2

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


@click.group(
    name="waymo",
    help="Utilities for using the Waymo Motion Dataset with SMARTS. See `scl waymo COMMAND --help` for further options.",
)
def waymo_cli():
    pass


@waymo_cli.command(
    name="overview", help="Display summary info for each scenario in the TFRecord file."
)
@click.argument(
    "tfrecord_file", type=click.Path(exists=True), metavar="<tfrecord_file>"
)
def overview(tfrecord_file: str):
    rows = []
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

        scenario_data = [
            scenario_id,
            len(scenario.timestamps_seconds),
            num_vehicles,
            num_pedestrians,
        ]
        rows.append(scenario_data)

    print(
        tabulate(
            rows,
            headers=[
                "Scenario ID",
                "Timestamps",
                "Vehicles",
                "Pedestrians",
            ],
        )
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


def _get_map_features(scenario: scenario_pb2.Scenario) -> Dict[str, List]:
    map_features = defaultdict(lambda: [])
    for i in range(len(scenario.map_features)):
        map_feature = scenario.map_features[i]
        key = map_feature.WhichOneof("feature_data")
        if key is not None:
            map_features[key].append((getattr(map_feature, key), str(map_feature.id)))
    return map_features


def _get_trajectories(
    scenario: scenario_pb2.Scenario,
) -> Dict[int, Dict[str, Any]]:
    num_steps = len(scenario.timestamps_seconds)
    trajectories = defaultdict(lambda: {"positions": [None] * num_steps})
    for i in range(len(scenario.tracks)):
        vehicle_id = scenario.tracks[i].id
        trajectories[vehicle_id]["is_ego"] = i == scenario.sdc_track_index
        trajectories[vehicle_id]["object_type"] = scenario.tracks[i].object_type
        for j in range(num_steps):
            obj_state = scenario.tracks[i].states[j]
            if obj_state.valid:
                trajectories[vehicle_id]["positions"][j] = (
                    obj_state.center_x,
                    obj_state.center_y,
                )
    return trajectories


def _plot_trajectories(
    trajectories: Dict[str, Any]
) -> Tuple[List[Line2D], List[Optional[Tuple[float, float]]]]:
    points, data = [], []

    # Need to plot something initially to get handles to the point objects,
    # so just use a valid point from the first trajectory
    first_traj = list(trajectories.values())[0]["positions"]
    ind = None
    for i in range(len(first_traj)):
        if first_traj[i] is not None:
            ind = i
            break
    assert ind is not None, "No valid point in first trajectory"
    x0 = first_traj[ind][0]
    y0 = first_traj[ind][1]
    for v_id, props in trajectories.items():
        xs = [p[0] for p in props["positions"] if p]
        ys = [p[1] for p in props["positions"] if p]
        if props["is_ego"]:
            (point,) = plt.plot(x0, y0, "c^")
        elif props["object_type"] == 1:
            (point,) = plt.plot(x0, y0, "k^")
        elif props["object_type"] == 2:
            (point,) = plt.plot(x0, y0, "md")
        elif props["object_type"] == 3:
            (point,) = plt.plot(x0, y0, "y*")
        else:
            (point,) = plt.plot(x0, y0, "k8")
        data.append((xs, ys))
        points.append(point)
    return points, data


@waymo_cli.command(
    name="preview", help="Plot the map and trajectories of the scenario."
)
@click.argument(
    "tfrecord_file", type=click.Path(exists=True), metavar="<tfrecord_file>"
)
@click.argument("scenario_id", type=str, metavar="<scenario_id>")
@click.option(
    "--animate_trajectories",
    is_flag=True,
    default=False,
    help="Animate the vehicle trajectories.",
)
@click.option(
    "--plot_trajectories",
    is_flag=True,
    default=False,
    help="Plot the initial positions of all vehicles with their IDs.",
)
def preview(
    tfrecord_file: str,
    scenario_id: str,
    animate_trajectories: bool,
    plot_trajectories: bool,
):
    source = f"{tfrecord_file}#{scenario_id}"
    scenario = WaymoMap.parse_source_to_scenario(source)
    fig = plt.figure()
    mng = plt.get_current_fig_manager()
    mng.resize(1000, 1000)
    map_features = _get_map_features(scenario)
    handles = MAP_HANDLES
    if plot_trajectories:
        handles.extend(TRAJECTORY_HANDLES)
        trajectories = _get_trajectories(scenario)
        for v_id, props in trajectories.items():
            xs = [p[0] for p in props["positions"] if p]
            ys = [p[1] for p in props["positions"] if p]
            plt.scatter(xs[0], ys[0], marker="o", c="blue")
            bbox_props = dict(boxstyle="square,pad=0.1", fc="white", ec=None)
            plt.text(xs[0] + 1, ys[0] + 1, f"{v_id}", bbox=bbox_props)
    elif animate_trajectories:
        handles.extend(TRAJECTORY_HANDLES)
        trajectories = _get_trajectories(scenario)
        points, data = _plot_trajectories(trajectories)

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


def _scenario_code(dataset_path, scenario_id):
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


@waymo_cli.command(
    name="export", help="Export the Waymo scenario to a SMARTS scenario."
)
@click.argument(
    "tfrecord_file", type=click.Path(exists=True), metavar="<tfrecord_file>"
)
@click.argument("scenario_id", type=str, metavar="<scenario_id>")
@click.argument(
    "export_folder", type=click.Path(exists=False), metavar="<export_folder>"
)
@click.option(
    "--generate-offline-data",
    is_flag=True,
    default=False,
    help="Extract observations and birds-eye view images for a selected set of vehicles for the scenario, for use in offline RL.",
)
def export(
    tfrecord_file: str,
    scenario_id: str,
    export_folder: str,
    generate_offline_data: bool,
):
    if not os.path.exists(export_folder):
        os.makedirs(export_folder)
    scenario_file = os.path.join(export_folder, "scenario.py")
    with open(scenario_file, "w") as f:
        f.write(_scenario_code(tfrecord_file, scenario_id))

    if generate_offline_data:
        pass  # TODO


waymo_cli.add_command(overview)
waymo_cli.add_command(preview)
waymo_cli.add_command(export)
