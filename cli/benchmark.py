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
import sys
import os
import subprocess
import click
from glob import glob
import smarts
from multiprocessing import Process, Semaphore, synchronize
from typing import List, Optional, Sequence
from smarts.benchmark import benchmark


@click.group(
    name="benchmark",
    help="Utilities for benchmarking the simulation performance. See `scl benchmark COMMAND --help` for further options.",
)
def benchmark_cli():
    pass

@click.command("run", help="Start all benchmarking.")
@click.argument("scenarios", nargs=-1, metavar="<scenarios>")
def build_all_scenarios_and_run(scenarios):
    # Build scenarios
    scenario_build_command = " ".join(
        ["scl scenario build-all"] + [f"{smarts.__path__[0]}/benchmark/{scenarios[0]}"]
    )
    subprocess.call(scenario_build_command, shell=True)
    # Run scenarios
    benchmark.main(
        scenarios=[f"{smarts.__path__[0]}/benchmark/{scenarios[0]}"],
    )

benchmark_cli.add_command(build_all_scenarios_and_run)
