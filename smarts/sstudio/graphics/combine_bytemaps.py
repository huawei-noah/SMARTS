# MIT License
#
# Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.
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
from enum import Enum
from pathlib import Path
from typing import Union

from smarts.sstudio.graphics.heightfield import HeightField


class Mode(Enum):
    multi = "multi"
    add = "add"
    sub = "sub"
    scale = "scale"
    max = "max"


def merge_bmps(bmp_1_file: Union[str, Path], bmp_2_file: Union[str, Path], output_file: Union[str, Path], mode=Mode.scale):
    hf1 = HeightField.load_image(bmp_1_file)
    hf2 = HeightField.load_image(bmp_2_file)

    method = hf1.multiply
    if mode is Mode.multi:
        method = hf1.multiply
    elif mode is Mode.sub:
        method = hf1.subtract
    elif mode is Mode.add:
        method = hf1.add
    elif mode is Mode.scale:
        method = hf1.scale_by
    elif mode is Mode.max:
        method = hf1.max

    hfo = method(hf2)

    with open(output_file, "wb") as f:
        hfo.write_image(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        os.path.basename(__file__),
        description="Utilities to merge bmp files.",
    )
    parser.add_argument("input_1", help="base file (*.bmp)", type=str)
    parser.add_argument("input_2", help="base file (*.bmp)", type=str)
    parser.add_argument("output", help="where to write the bytemap file", type=str)
    parser.add_argument(
        "--mode",
        help=f"the actions: {[m for m in Mode.__members__]}",
        type=str,
        default="scale",
    )
    args = parser.parse_args()

    merge_bmps(args.input_1, args.input_2, args.output, Mode(args.mode))
