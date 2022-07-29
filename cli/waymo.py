from collections import defaultdict
import os
from typing import Dict, List, Optional, Tuple

import click
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
) -> Dict[int, List[Optional[Tuple[float, float]]]]:
    num_steps = len(scenario.timestamps_seconds)
    trajectories = defaultdict(lambda: [None] * num_steps)
    for i in range(len(scenario.tracks)):
        vehicle_id = scenario.tracks[i].id
        for j in range(num_steps):
            obj_state = scenario.tracks[i].states[j]
            if obj_state.valid:
                trajectories[vehicle_id][j] = (obj_state.center_x, obj_state.center_y)
    return trajectories


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
    if plot_trajectories:
        trajectories = _get_trajectories(scenario)
        for v_id, positions in trajectories.items():
            xs = [p[0] for p in positions if p]
            ys = [p[1] for p in positions if p]
            plt.scatter(xs[0], ys[0], marker="o", c="blue")
            bbox_props = dict(boxstyle="square,pad=0.1", fc="white", ec=None)
            plt.text(xs[0] + 1, ys[0] + 1, f"{v_id}", bbox=bbox_props)
    if animate_trajectories:
        pass  # TODO
    _plot_map_features(map_features)
    plt.title(f"Scenario {scenario_id}")
    plt.legend(handles=MAP_HANDLES)
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
