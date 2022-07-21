from typing import Dict, List
import click
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from smarts.core.utils.file import read_tfrecord_file
from smarts.core.waymo_map import WaymoMap
from tabulate import tabulate
from waymo_open_dataset.protos import scenario_pb2

from smarts.sstudio.types import MapSpec


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


def _plot_map_features(map_features: Dict, feature_ids: List[str]) -> List[Line2D]:
    """Plot the map features with some feature_ids highlighted and return extended legend handles"""
    handles = []
    for lane in map_features["lane"]:
        pts = np.array([[p.x, p.y] for p in lane[0].polyline])
        if lane[1] in feature_ids:
            plt.plot(pts[:, 0], pts[:, 1], linestyle=":", color="blue", linewidth=2.0)
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
        pts = np.array([[p.x, p.y] for p in road_line[0].polyline])
        xs, ys = pts[:, 0], pts[:, 1]
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
        pts = np.array([[p.x, p.y] for p in road_edge[0].polyline])
        xs, ys = pts[:, 0], pts[:, 1]
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
        pts = np.array([[p.x, p.y] for p in crosswalk[0].polyline])
        xs, ys = pts[:, 0], pts[:, 1]
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
        pts = np.array([[p.x, p.y] for p in speed_bump[0].polyline])
        xs, ys = pts[:, 0], pts[:, 1]
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


@waymo_cli.command(
    name="preview", help="Plot the map and trajectories of the scenario."
)
@click.argument(
    "tfrecord_file", type=click.Path(exists=True), metavar="<tfrecord_file>"
)
@click.argument("scenario_id", type=str, metavar="<scenario_id>")
@click.option(
    "--animate",
    is_flag=True,
    default=False,
    help="Animate the vehicle trajectories instead of plotting them.",
)
def preview(tfrecord_file: str, scenario_id: str, animate: bool):
    spec = MapSpec(f"{tfrecord_file}#{scenario_id}")
    map = WaymoMap.from_spec(spec)
    assert map is not None, "Invalid tfrecord file or scenario id"
    scenario = WaymoMap.parse_source_to_scenario(spec.source)

    # fig = plt.figure()
    # highlighted_handles = _plot_map_features(map_features, [])
    # plt.title(f"Scenario {scenario_id}")


waymo_cli.add_command(overview)
