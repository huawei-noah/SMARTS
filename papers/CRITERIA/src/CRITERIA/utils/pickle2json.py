# MIT License
#
# Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
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
from pathlib import Path

import json
import numpy as np

from .common import get_file_formatter


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyArrayEncoder, self).default(obj)


def pickle2json(file_name, in_mode: str = "rb", out_mode: str = "wt"):
    file_path = Path(file_name)
    formatter = get_file_formatter("pickle")
    with open(file_path, in_mode) as f:
        data = formatter.formatter.load(f)

    file_formatter = get_file_formatter("json")
    out_file_path = file_path.with_suffix(f".{file_formatter.ext}")
    with open(out_file_path, out_mode) as f:
        file_formatter.formatter.dump(data, f, cls=NumpyArrayEncoder)
    print(f'"{out_file_path.__str__()}"')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(Path(__file__).stem)
    parser.add_argument(
        "pickle_file",
        help="The pickle file to convert.",
        type=str,
    )
    args = parser.parse_args()

    pickle2json(args.pickle_file)
