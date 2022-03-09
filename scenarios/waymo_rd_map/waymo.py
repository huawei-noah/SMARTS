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

from typing import Dict, List, Tuple, Union

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
    map_features = {"lane": [], "road_line": [], "road_edge": [], "stop_sign": [], "crosswalk": [], "speed_bump": []}
    for i in range(len(scenario.map_features)):
        map_feature = scenario.map_features[i]
        key = map_feature.WhichOneof("feature_data")
        if key is not None:
            map_features[key].append(getattr(map_feature, key))
            if key == "lane":
                lanes.append((getattr(map_feature, key), map_feature.id))
    # tls_lanes = get_traffic_light_lanes(scenario)
    return map_features, lanes


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


def dump_plots(outdir: str, path: str) -> List[str]:
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
        out_path = os.path.join(outdir, filename)
        fig = plt.gcf()
        # w, h = mng.window.maxsize()
        dpi = 100
        fig.set_size_inches(w / dpi, h / dpi)
        print(f"Saving {out_path}")
        fig.savefig(out_path, dpi=100)
        plt.close("all")
    return scenarios


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="waymo.py",
        description="Extract map data from Waymo dataset and convert to SUMO.",
    )
    parser.add_argument("file", help="TFRecord file")
    parser.add_argument(
        "--outdir", help="output directory for screenshots", type=str
    )
    parser.add_argument(
        "--gen",
        help="generate sumo map",
        type=str,
        nargs=1,
        metavar="SCENARIO_ID",
    )
    parser.add_argument(
        "--plot",
        help="plot scenario map",
        type=str,
        nargs=1,
        metavar="SCENARIO_ID",
    )
    args = parser.parse_args()

    if args.outdir:
        scenario_hashes = dump_plots(args.outdir, args.file)
        print(scenario_hashes)
    else:
        plot(args.file, args.plot[0])
