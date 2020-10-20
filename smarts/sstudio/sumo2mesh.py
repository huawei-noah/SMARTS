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

from smarts.core.sumo_road_network import SumoRoadNetwork


def generate_glb_from_sumo_network(sumo_net_file, out_glb_file):
    road_network = SumoRoadNetwork.from_file(net_file=sumo_net_file)
    glb = road_network.build_glb(scale=1000)
    glb.write_glb(out_glb_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "sumo2mesh.py",
        description="Utility to export sumo road networks to mesh files.",
    )
    parser.add_argument("net", help="sumo net file (*.net.xml)", type=str)
    parser.add_argument("output_path", help="where to write the mesh file", type=str)
    args = parser.parse_args()

    generate_glb_from_sumo_network(args.net, args.output_path)
