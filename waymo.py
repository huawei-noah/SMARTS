import argparse
import io
import os
import subprocess
from typing import List, Tuple
import xml.dom.minidom
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.animation import FuncAnimation
from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.protos import map_pb2
from smarts.sstudio.genhistories import Waymo

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def convert_polyline(polyline) -> Tuple[List[float], List[float]]:
    tuples = [(p.x, p.y) for p in polyline]
    xs, ys = zip(*tuples)
    return xs, ys

def read_trajectory_data(path, scenario_index):
    dataset_spec = {"input_path": path, "scenario_index": scenario_index}
    dataset = Waymo(dataset_spec, None)

    trajectories = {}
    agent_id = None
    ego_id = None
    for row in dataset.rows:
        if agent_id != row['vehicle_id']:
            agent_id = row['vehicle_id']
            trajectories[agent_id] = ([], [])
            if row['is_ego_vehicle'] == 1:
                ego_id = agent_id
        trajectories[agent_id][0].append(row['position_x'])
        trajectories[agent_id][1].append(row['position_y'])
    return trajectories, ego_id

def read_map_data(path, scenario_index):
    dataset = tf.data.TFRecordDataset(
        path, compression_type=""
    )
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

def shape_str(xs, ys):
    result = ""
    for x, y in zip(xs, ys):
        result += f"{x},{y} "
    return result[:-1]

def generate_sumo_map(path, scenario_index):
    def make_counter():
        i = 0
        def f():
            nonlocal i
            i += 1
            return i
        return f

    edge_counter = make_counter()
    node_counter = make_counter()
    nodes_root = ET.Element("nodes")
    edges_root = ET.Element("edges")

    def add_segment(xs, ys):
        nonlocal node_counter, edge_counter
        nonlocal nodes_root, edges_root

        start_node_id = f"node-{node_counter()}"
        start_node = ET.SubElement(nodes_root, "node")
        start_node.set("id", start_node_id)
        start_node.set("type", "priority")
        start_node.set("x", str(xs[0]))
        start_node.set("y", str(ys[0]))

        end_node_id = f"node-{node_counter()}"
        end_node = ET.SubElement(nodes_root, "node")
        end_node.set("id", end_node_id)
        end_node.set("type", "priority")
        end_node.set("x", str(xs[-1]))
        end_node.set("y", str(ys[-1]))

        edge = ET.SubElement(edges_root, "edge")
        edge.set("id", f"edge-{edge_counter()}")
        edge.set("from", start_node_id)
        edge.set("to", end_node_id)
        edge.set("priority", str(1))
        edge.set("numLanes", str(1))
        edge.set("speed", str(11.0))
        lane = ET.SubElement(edge, "lane")
        lane.set("index", str(0))
        lane.set("width", str(4))
        lane.set("shape", shape_str(xs, ys))

    scenario_id, map_features = read_map_data(path, scenario_index)
    nodes_path = f"nodes-{scenario_id}.nod.xml"
    edges_path = f"edges-{scenario_id}.edg.xml"
    net_path = f"net-{scenario_id}.net.xml"

    lanes = [convert_polyline(lane.polyline) for lane in map_features["lane"]]
    # lanes = list(filter(lambda lane: max(lane[1]) > 8150, lanes))

    # Build XML
    for lane in lanes:
        add_segment(*lane)

    # Write XML
    edges_xml = xml.dom.minidom.parseString(ET.tostring(edges_root)).toprettyxml()
    nodes_xml = xml.dom.minidom.parseString(ET.tostring(nodes_root)).toprettyxml()
    with open(edges_path, "w") as f:
        f.write(edges_xml)
    with open(nodes_path, "w") as f:
        f.write(nodes_xml)

    # Generate netfile with netconvert
    print(f"Generating SUMO map file: {net_path}")
    proc = subprocess.Popen([
            'netconvert',
            f'--node-files={nodes_path}',
            f'--edge-files={edges_path}',
            f'--output-file={net_path}',
            "--offset.disable-normalization",
            # "--geometry.split"
            # '--junctions.join'
        ],
        stdout=subprocess.PIPE
    )
    for line in io.TextIOWrapper(proc.stdout, encoding="utf-8"):
        print(line.rstrip())

def safe_print(obj, name):
    if hasattr(obj, name):
        print(getattr(obj, name))

def plot_map(map_features):
    lanes = [convert_polyline(lane.polyline) for lane in map_features["lane"]]
    # lanes = list(filter(lambda lane: max(lane[1]) > 8150, lanes))
    for xs, ys in lanes:
        plt.plot(xs, ys, linestyle=':', color='gray')
    for road_line in map_features["road_line"]:
        xs, ys = convert_polyline(road_line.polyline)
        if road_line.type in [1, 4, 5]:
            plt.plot(xs, ys, 'y--')
        else:
            plt.plot(xs, ys, 'y-')
    for road_edge in map_features["road_edge"]:
        xs, ys = convert_polyline(road_edge.polyline)
        plt.plot(xs, ys, 'k-')
    # for crosswalk in map_features["crosswalk"]:
    #     xs, ys = convert_polyline(crosswalk.polygon)
    #     plt.plot(xs, ys, 'k--')
    # for speed_bump in map_features["speed_bump"]:
    #     xs, ys = convert_polyline(speed_bump.polygon)
    #     plt.plot(xs, ys, 'k:')
    for stop_sign in map_features["stop_sign"]:
        plt.scatter(stop_sign.position.x, stop_sign.position.y, marker='o', c='#ff0000', alpha=1)

def plot(path, scenario_index):
    # Get data
    trajectories, ego_id = read_trajectory_data(path, scenario_index)
    scenario_id, map_features = read_map_data(path, scenario_index)

    # Plot map and trajectories
    fig, ax = plt.subplots()
    ax.set_title(f"Scenario {scenario_id})")
    
    plot_map(map_features)
    
    # for k, v in trajectories.items():
    #     plt.scatter(v[0], v[1], marker='.')
    
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.show()

def animate(path, scenario_index, screenshot=False, outdir=None):
    # Get data
    trajectories, ego_id = read_trajectory_data(path, scenario_index)
    scenario_id, map_features = read_map_data(path, scenario_index)

    # Plot map and trajectories
    fig, ax = plt.subplots()
    ax.set_title(f"Scenario {scenario_id}")

    plot_map(map_features)

    max_len = 0
    data, points = [], []
    for k, v in trajectories.items():
        xs, ys = v[0], v[1]
        if len(xs) > max_len:
            max_len = len(xs)
        if k == ego_id:
            point, = plt.plot(xs[0], ys[0], 'cs')
        else:    
            point, = plt.plot(xs[0], ys[0], 'ks')
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
    else:
        filename = f'scenario-{scenario_index}-{scenario_id}.png'
        outpath = os.path.join(outdir, filename)
        fig = plt.gcf()
        w, h = mng.window.maxsize()
        dpi = 100
        fig.set_size_inches(w/dpi, h/dpi)
        print(f"Saving {outpath}")
        fig.savefig(outpath, dpi=100)
        plt.close('all')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="convert_waymo_map.py",
        description="Extract map data from Waymo dataset and convert to SUMO.",
    )
    parser.add_argument("file", help="TFRecord file")
    parser.add_argument(
        "--outdir",
        help="output directory for screenshots",
        type=str,
        nargs='?'
    )
    parser.add_argument(
        "--gen",
        help="generate sumo map",
        type=int,
        default=0,
        nargs='?',
        metavar='SCENARIO_INDEX'
    )
    parser.add_argument(
        "--plot",
        help="plot scenario map",
        type=int,
        default=0,
        nargs='?',
        metavar='SCENARIO_INDEX'
    )
    parser.add_argument(
        "--animate",
        help="plot scenario map and animate trajectories",
        type=int,
        default=0,
        nargs='?',
        metavar='SCENARIO_INDEX'
    )
    args = parser.parse_args()

    if args.outdir:
        for i in range(79):
            animate(args.file, i, screenshot=True, outdir=args.outdir)
    else:
        if args.gen:
            generate_sumo_map(args.file, args.gen)
        elif args.plot:
            plot(args.file, args.plot)
        elif args.animate:
            animate(args.file, args.animate)
