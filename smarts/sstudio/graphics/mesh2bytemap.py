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
# FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
import argparse
import os

import numpy as np

from smarts.core.coordinates import Dimensions, Pose
from smarts.core.sensor import DrivableAreaGridMapSensor
from smarts.core.vehicle_state import VehicleState
from smarts.p3d.renderer import Renderer
from smarts.sstudio.graphics.heightfield import HeightField


def generate_bytemap_from_glb_file(
    glb_file: str, out_bytemap_file: str, padding: int, inverted=False
):
    """Creates a geometry file from a sumo map file."""
    renderer = Renderer("r")
    bounds = renderer.load_road_map(glb_file)
    size = (*(bounds.max - bounds.min),)
    vs = VehicleState(
        "a",
        pose=Pose(np.array((*bounds.center,)), np.array([0, 0, 0, 1])),
        dimensions=Dimensions(1, 1, 1),
    )
    camera = DrivableAreaGridMapSensor(
        vehicle_state=vs,
        width=int(size[0]) + padding,
        height=int(size[1]) + padding,
        resolution=1,
        renderer=renderer,
    )
    renderer.render()
    image = camera(renderer)
    data = image.data
    if inverted:
        hf = HeightField(data, (1, 1))
        data = hf.inverted().data

    from PIL import Image

    im = Image.fromarray(data.squeeze(), "L")
    im.save(out_bytemap_file)
    im.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        os.path.basename(__file__),
        description="Utility to export mesh files to bytemap.",
    )
    parser.add_argument("mesh", help="mesh file (*.glb)", type=str)
    parser.add_argument("output_path", help="where to write the bytemap file", type=str)
    parser.add_argument("--padding", help="the padding pixels", type=int, default=0)
    parser.add_argument(
        "--inverted", help="invert the map surface", action="store_true"
    )
    args = parser.parse_args()

    generate_bytemap_from_glb_file(
        args.mesh, args.output_path, padding=args.padding, inverted=args.inverted
    )
