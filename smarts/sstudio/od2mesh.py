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
import argparse
from pathlib import Path

from smarts.core.opendrive_road_network import OpenDriveRoadNetwork


def generate_glb_from_opendrive_network(scenario):
    scenario_root = Path(scenario)
    map_xodr = str(scenario_root / "UC_Motorway-Exit-Entry.xodr")
    road_map = OpenDriveRoadNetwork.from_file(map_xodr)
    assert isinstance(road_map, OpenDriveRoadNetwork)
    map_glb = map_xodr + "_map.glb"
    road_map.to_glb(map_glb)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "od2mesh.py",
        description="Utility to export opendrive road networks to mesh files.",
    )
    parser.add_argument(
        "scenario_path", help="path to opendrive xodr file (*.xodr)", type=str
    )
    args = parser.parse_args()

    generate_glb_from_opendrive_network(args.scenario_path)
