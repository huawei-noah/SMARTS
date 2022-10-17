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
import os
import subprocess
import click
from glob import glob


@click.group(
    name="benchmark",
    help="Utilities for benchmarking the simulation performance. See `scl benchmark COMMAND --help` for further options.",
)


def benchmark_cli():
    pass


@benchmark_cli.command(
    name="start", help="Start selected benchmarking."
)
@click.option("--all", help="run all benchmarks.")
@click.argument("scenario", type=click.Path(exists=True), metavar="<scenario>")
def build_scenario_and_run(scenario):
    subprocess.run(
        [
            "scl",
            "scenario",
            "build",
            "--clean",
            f"{scenario}"
        ]
    )
    subprocess.run(
        [
            "scl",
            "run",
            "examples/egoless.py",
            f"{scenario}"
        ]
    )

def build_all_scenarios_and_run(scenario):
    subprocess.run(
        [
            "scl",
            "scenario",
            "build-all",
            "--clean",
            "scenarios/benchmark/"
        ]
    )

    all_scenarios = glob('./scenarios/benchmark/*/*/', recursive=True)
    for scenario in all_scenarios:
        subprocess.run(
            [
            "scl",
            "run",
            "examples/egoless.py",
            f"{scenario}",
            ]
        ) 
benchmark_cli.add_command(build_scenario_and_run)


# `scl benchmark --all <path/to/scenarios>`` # Run all benchmarks in <path/to/scenarios> folder.
# `scl benchmark <path/to/scenario>` # Run the selected one benchmark in the <path/to/scenario> folder.