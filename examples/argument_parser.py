# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
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
from typing import Optional


def default_argument_parser(program: Optional[str] = None):
    """This factory method returns a vanilla `argparse.ArgumentParser` with the
    minimum subset of arguments that should be supported.

    You can extend it with more `parser.add_argument(...)` calls or obtain the
    arguments via `parser.parse_args()`.
    """
    if not program:
        from pathlib import Path

        program = Path(__file__).stem

    parser = argparse.ArgumentParser(program)
    parser.add_argument(
        "scenarios",
        help="A list of scenarios. Each element can be either the scenario to"
        "run or a directory of scenarios to sample from. See `scenarios/`"
        "folder for some samples you can use.",
        type=str,
        nargs="*",
    )
    parser.add_argument(
        "--episodes",
        help="The number of episodes to run the simulation for.",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--headless", help="Run the simulation in headless mode.", action="store_true"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sim-name",
        help="Simulation name.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--sumo-port", help="Run SUMO with a specified port.", type=int, default=None
    )
    return parser
