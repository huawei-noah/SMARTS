#%%
import argparse
import os
from typing import List, Tuple
import matplotlib

matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.animation import FuncAnimation, FFMpegWriter
from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.protos import map_pb2
from smarts.sstudio.genhistories import Waymo

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def convert_polyline(polyline) -> Tuple[List[float], List[float]]:
    tuples = [(p.x, p.y) for p in polyline]
    xs, ys = zip(*tuples)
    return xs, ys


def read_trajectory_data(path, scenario_index):
    dataset_spec = {"input_path": path}
    dataset = Waymo(dataset_spec, None)

    trajectories = {}
    agent_id = None
    ego_id = None
    for row in dataset.rows:
        if agent_id != row["vehicle_id"]:
            agent_id = row["vehicle_id"]
            trajectories[agent_id] = ([], [])
            if row["is_ego_vehicle"] == 1:
                ego_id = agent_id
        trajectories[agent_id][0].append(row["position_x"])
        trajectories[agent_id][1].append(row["position_y"])
    return trajectories, ego_id


def read_map_data(path, scenario_index):
    dataset = tf.data.TFRecordDataset(path, compression_type="")
    scenario_list = list(dataset.as_numpy_iterator())
    scenario_data = scenario_list[scenario_index]
    scenario = scenario_pb2.Scenario()
    scenario.ParseFromString(bytearray(scenario_data))

    map_features = {}
    map_features["lane"] = []
    map_features["road_line"] = []
    map_features["road_edge"] = []
    map_features["stop_sign"] = []
    map_features["crosswalk"] = []
    map_features["speed_bump"] = []
    for i in range(len(scenario.map_features)):
        map_feature = scenario.map_features[i]
        key = map_feature.WhichOneof("feature_data")
        if key is not None:
            map_features[key].append(getattr(map_feature, key))

    return scenario.scenario_id, map_features


def plot(path, scenario_index):
    trajectories, ego_id = read_trajectory_data(path, scenario_index)
    scenario_id, map_features = read_map_data(path, scenario_index)

    fig, ax = plt.subplots()
    ax.set_title(f"Scenario {scenario_id})")

    # Plot map
    for lane in map_features["lane"]:
        xs, ys = convert_polyline(lane.polyline)
        plt.plot(xs, ys, linestyle=":", color="gray")
    for road_line in map_features["road_line"]:
        xs, ys = convert_polyline(road_line.polyline)
        plt.plot(xs, ys, "y-")
    for road_edge in map_features["road_edge"]:
        xs, ys = convert_polyline(road_edge.polyline)
        plt.plot(xs, ys, "k-")
    for crosswalk in map_features["crosswalk"]:
        xs, ys = convert_polyline(crosswalk.polygon)
        plt.plot(xs, ys, "k--")
    for speed_bump in map_features["speed_bump"]:
        xs, ys = convert_polyline(speed_bump.polygon)
        plt.plot(xs, ys, "k:")
    for stop_sign in map_features["stop_sign"]:
        plt.scatter(
            stop_sign.position.x, stop_sign.position.y, marker="o", c="#ff0000", alpha=1
        )

    # Plot trajectories
    for k, v in trajectories.items():
        plt.scatter(v[0], v[1], marker=".")

    plt.show()


def animate(path, scenario_index, screenshot=False, outdir=None):
    trajectories, ego_id = read_trajectory_data(path, scenario_index)
    scenario_id, map_features = read_map_data(path, scenario_index)

    fig, ax = plt.subplots()
    ax.set_title(f"Scenario {scenario_id}")

    # Plot map
    objs = []
    for lane in map_features["lane"]:
        xs, ys = convert_polyline(lane.polyline)
        obj = plt.plot(xs, ys, linestyle=":", color="gray")
        objs.append(obj)
    for road_line in map_features["road_line"]:
        xs, ys = convert_polyline(road_line.polyline)
        obj = plt.plot(xs, ys, "y-")
        objs.append(obj)
    for road_edge in map_features["road_edge"]:
        xs, ys = convert_polyline(road_edge.polyline)
        obj = plt.plot(xs, ys, "k-")
        objs.append(obj)
    for crosswalk in map_features["crosswalk"]:
        xs, ys = convert_polyline(crosswalk.polygon)
        obj = plt.plot(xs, ys, "k--")
        objs.append(obj)
    for speed_bump in map_features["speed_bump"]:
        xs, ys = convert_polyline(speed_bump.polygon)
        obj = plt.plot(xs, ys, "k:")
        objs.append(obj)
    for stop_sign in map_features["stop_sign"]:
        obj = plt.scatter(
            stop_sign.position.x, stop_sign.position.y, marker="o", c="#ff0000", alpha=1
        )
        objs.append(obj)

    # Plot trajectories
    max_len = 0
    data, points = [], []
    for k, v in trajectories.items():
        xs, ys = v[0], v[1]
        if len(xs) > max_len:
            max_len = len(xs)
        if k == ego_id:
            (point,) = plt.plot(xs[0], ys[0], "cs")
        else:
            (point,) = plt.plot(xs[0], ys[0], "ks")
        data.append((xs, ys))
        points.append(point)

    def update(i):
        drawn_pts = []
        for (xs, ys), point in zip(data, points):
            if i < len(xs):
                point.set_data(xs[i], ys[i])
                drawn_pts.append(point)
        return drawn_pts

    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    if not screenshot:
        anim = FuncAnimation(fig, update, frames=max_len, blit=True, interval=100)
        plt.show()
        # writer = FFMpegWriter(fps=15)
        # anim.save(f'{scenario_id}.mp4', writer=writer)
    else:
        filename = f"scenario-{scenario_index}-{scenario_id}.png"
        outpath = os.path.join(outdir, filename)
        fig = plt.gcf()
        w, h = mng.window.maxsize()
        dpi = 100
        fig.set_size_inches(w / dpi, h / dpi)
        print(f"Saving {outpath}")
        fig.savefig(outpath, dpi=100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="convert_waymo_map.py",
        description="Extract map data from Waymo dataset and convert to SUMO.",
    )
    parser.add_argument("path")
    parser.add_argument("--foo", nargs="?", help="foo help")
    parser.add_argument("--outdir", help="output directory for screenshots", nargs="?")
    args = parser.parse_args()

    if args.outdir:
        for i in range(79):
            animate(args.path, i, screenshot=True, outdir=args.outdir)
    else:
        plot(args.path, 1)
    animate(args.path, 3, screenshot=False)
# %%
