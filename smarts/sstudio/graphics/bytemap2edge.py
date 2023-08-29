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

import numpy as np

from smarts.sstudio.graphics.heightfield import HeightField


def generate_edge_bytemap_from_bytemap(bytemap_file, out_edge_bytemap_file):
    hf = HeightField.load_image(bytemap_file)
    bhf = hf.apply_kernel(
        kernel=np.array(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 0.5, 0.0, 0.5, 1.0, 1.0],
                [1.0, 0.0, -1.0, -1.0, -1.0, 0.0, 1.0],
                [0.5, 0.0, -1.0, -256, -1.0, 0.0, 0.5],
                [1.0, 0.0, -1.0, -1.0, -1.0, 0.0, 1.0],
                [1.0, 0.5, 0.5, 0.0, 0.5, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ]
        ),
        min_val=0,
        max_val=255,
    )
    bhf.write_image(out_edge_bytemap_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "bytemap2edge.py",
        description="Utility to convert a bytemap to an edge bytemap.",
    )
    parser.add_argument("bytemap", help="bytemap file (*.bmp)", type=str)
    parser.add_argument(
        "output_path", help="where to write the edge bytemap file", type=str
    )
    args = parser.parse_args()

    generate_edge_bytemap_from_bytemap(args.bytemap, args.output_path)
