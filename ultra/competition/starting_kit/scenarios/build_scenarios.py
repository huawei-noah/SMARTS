# MIT License
#
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
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

from ultra.scenarios.generate_scenarios import build_scenarios


if __name__ == "__main__":
    parser = argparse.ArgumentParser("build-scenarios")
    parser.add_argument(
        "--task",
        help="The name of the task used to describe the scenarios.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--level",
        help="The level of the config from which the scenarios will be built.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--save-dir",
        help="Where to save the created scenarios.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--root-dir",
        help="The directory containing the task directories.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--pool-dir",
        help="The directory containing the map files.",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    build_scenarios(
        task=args.task,
        level_name=args.level,
        stopwatcher_behavior=None,
        stopwatcher_route=None,
        save_dir=args.save_dir,
        root_path=args.root_dir,
        pool_dir=args.pool_dir,
    )
