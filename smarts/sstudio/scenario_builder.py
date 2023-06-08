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

from smarts.core import seed as smarts_seed


def _build(scenario_py: str, seed: int):
    # Make sure all the seed values are consistent before running the scenario script
    smarts_seed(seed)

    # Execute the scenario script, using the current globals that were set by the seed value
    with open(scenario_py, "rb") as source_file:
        code = compile(source_file.read(), scenario_py, "exec")
        exec(code, {"__file__": scenario_py})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario_py", help="Path to the scenario.py file.", type=str)
    parser.add_argument(
        "seed",
        help="Seed that will be set before executing the scenario script.",
        type=int,
    )
    args = parser.parse_args()
    _build(args.scenario_py, args.seed)
