import argparse
import io
import os
import subprocess
import xml.dom.minidom
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple

from smarts.sstudio.genhistories import Waymo
from waymo_open_dataset.protos import scenario_pb2

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def read_map_data(path, scenario_id):
    scenario = None
    dataset = Waymo.read_dataset(path)
    for record in dataset:
        parsed_scenario = scenario_pb2.Scenario()
        parsed_scenario.ParseFromString(bytearray(record))
        if parsed_scenario.scenario_id == scenario_id:
            scenario = parsed_scenario
            break

    if scenario is None:
        errmsg = f"Dataset file does not contain scenario with id: {scenario_id}"
        raise ValueError(errmsg)

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


def convert_polyline(polyline) -> Tuple[List[float], List[float]]:
    tuples = [(p.x, p.y) for p in polyline]
    xs, ys = zip(*tuples)
    return xs, ys


def shape_str(xs, ys):
    result = ""
    for x, y in zip(xs, ys):
        result += f"{x},{y} "
    return result[:-1]


def generate_sumo_map(path, scenario_id):
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
        lane.set("width", str(5))
        lane.set("shape", shape_str(xs, ys))

    scenario_id, map_features = read_map_data(path, scenario_id)
    base_dir = Path(__file__).absolute().parent
    nodes_path = base_dir / f"nodes-{scenario_id}.nod.xml"
    edges_path = base_dir / f"edges-{scenario_id}.edg.xml"
    net_path = base_dir / f"map-{scenario_id}.net.xml"

    lanes = [convert_polyline(lane.polyline) for lane in map_features["lane"]]

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
    proc = subprocess.Popen(
        [
            "netconvert",
            f"--node-files={nodes_path}",
            f"--edge-files={edges_path}",
            f"--output-file={net_path}",
            "--offset.disable-normalization",
            "--no-internal-links=false",
        ],
        stdout=subprocess.PIPE,
    )
    for line in io.TextIOWrapper(proc.stdout, encoding="utf-8"):
        print(line.rstrip())

    # Clean up intermediate files
    os.remove(nodes_path)
    os.remove(edges_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="gen_sumo_map.py",
        description="Extract map data from a Waymo motion dataset scenario and generate a SUMO map.",
    )
    parser.add_argument("file", help="TFRecord file")
    parser.add_argument("id", help="ID of the scenario")
    args = parser.parse_args()
    generate_sumo_map(args.file, args.id)
