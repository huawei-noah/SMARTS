import argparse
import io
import os
import subprocess
import xml.dom.minidom
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from smarts.sstudio.genhistories import Waymo
from waymo_open_dataset.protos import map_pb2, scenario_pb2


class SumoMapGenerator:
    def __init__(self):
        self.nodes_root = None
        self.edges_root = None

    @staticmethod
    def _read_map_data(path: str, scenario_id: str) -> Dict:
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

        return map_features

    @staticmethod
    def _convert_polyline(
        polyline: map_pb2.MapPoint,
    ) -> Tuple[List[float], List[float]]:
        tuples = [(p.x, p.y) for p in polyline]
        xs, ys = zip(*tuples)
        return xs, ys

    @staticmethod
    def _shape_str(xs: List[float], ys: List[float]) -> str:
        result = ""
        for x, y in zip(xs, ys):
            result += f"{x},{y} "
        return result[:-1]

    @staticmethod
    def _make_counter() -> Callable[[], int]:
        i = 0

        def f() -> int:
            nonlocal i
            i += 1
            return i

        return f

    def _create_node(self, node_id: str, x: float, y: float):
        node = ET.SubElement(self.nodes_root, "node")
        node.set("id", node_id)
        node.set("type", "priority")
        node.set("x", str(x))
        node.set("y", str(y))

    def _create_edge(
        self,
        edge_id: str,
        start_id: str,
        end_id: str,
        shape_str: Tuple[List[float], List[float]],
        width: float = 5,
    ):
        edge = ET.SubElement(self.edges_root, "edge")
        edge.set("id", edge_id)
        edge.set("from", start_id)
        edge.set("to", end_id)
        edge.set("priority", str(1))
        edge.set("numLanes", str(1))

        lane = ET.SubElement(edge, "lane")
        lane.set("index", str(0))
        lane.set("width", str(width))
        lane.set("shape", shape_str)

    def generate(self, path: str, scenario_id: str):
        edge_counter = SumoMapGenerator._make_counter()
        node_counter = SumoMapGenerator._make_counter()
        self.nodes_root = ET.Element("nodes")
        self.edges_root = ET.Element("edges")

        map_features = SumoMapGenerator._read_map_data(path, scenario_id)
        base_dir = Path(__file__).absolute().parent
        nodes_path = base_dir / f"nodes-{scenario_id}.nod.xml"
        edges_path = base_dir / f"edges-{scenario_id}.edg.xml"
        net_path = base_dir / f"map-{scenario_id}.net.xml"

        lanes = [
            SumoMapGenerator._convert_polyline(lane.polyline)
            for lane in map_features["lane"]
        ]

        # Build XML
        for xs, ys in lanes:
            start_id = f"node-{node_counter()}"
            end_id = f"node-{node_counter()}"
            edge_id = f"edge-{edge_counter()}"
            self._create_node(start_id, xs[0], ys[0])
            self._create_node(end_id, xs[-1], ys[-1])
            self._create_edge(
                edge_id, start_id, end_id, SumoMapGenerator._shape_str(xs, ys)
            )

        # Write XML
        edges_xml = xml.dom.minidom.parseString(
            ET.tostring(self.edges_root)
        ).toprettyxml()
        nodes_xml = xml.dom.minidom.parseString(
            ET.tostring(self.nodes_root)
        ).toprettyxml()
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

    map_generator = SumoMapGenerator()
    map_generator.generate(args.file, args.id)
