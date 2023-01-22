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

from pathlib import Path
from typing import Optional

import click


@click.group(
    name="benchmark",
    help="Utilities for running integrated ML benchmarks. See `scl benchmark COMMAND --help` for further options.",
)
def benchmark_cli():
    pass


@click.command(
    "run",
)
@click.argument("benchmark_id", nargs=1, metavar="<benchmark_id>")
@click.argument("agent_config", nargs=1, metavar="<agent_config>")
@click.option(
    "--debug-log",
    is_flag=True,
    default=False,
    help="Log the benchmark in stdout.",
)
@click.option(
    "--benchmark-listing",
    type=str,
    default=None,
    help="Directs to a different listing file.",
)
@click.option(
    "--auto-install",
    is_flag=True,
    default=False,
    help="Attempt to auto install requirements.",
)
def run(
    benchmark_id: str,
    agent_config: str,
    debug_log: bool,
    benchmark_listing: Optional[str],
    auto_install: bool,
):
    """This runs a given benchmark.

    Use `scl benchmark list` to see the available benchmarks.

    \b
    <benchmark_id> is formatted like BENCHMARK_NAME==BENCHMARK_VERSION.
    <agent_config> is the path to an agent configuration file.

    An example use: `scl benchmark run --auto-install driving_smarts==0.0 ./baselines/driving_smarts/v0/agent_config.yaml`
    """
    from smarts.benchmark import BENCHMARK_LISTING_FILE, run_benchmark

    benchmark_id, _, benchmark_version = benchmark_id.partition("==")

    run_benchmark(
        benchmark_id,
        float(benchmark_version) if benchmark_version else None,
        Path(agent_config),
        Path(benchmark_listing)
        if benchmark_listing is not None
        else BENCHMARK_LISTING_FILE,
        debug_log,
        auto_install=auto_install,
    )


@click.command("list")
@click.option(
    "--benchmark-listing",
    type=str,
    default=None,
    help="Directs to a different listing file.",
)
def list_benchmarks(benchmark_listing: Optional[str]):
    """Lists available benchmarks that can be used for `scl benchmark run`."""
    from smarts.benchmark import BENCHMARK_LISTING_FILE
    from smarts.benchmark import list_benchmarks as _list_benchmarks

    benchmarks = _list_benchmarks(
        Path(benchmark_listing)
        if benchmark_listing is not None
        else BENCHMARK_LISTING_FILE,
    )["benchmarks"]

    print("BENCHMARK_NAME".ljust(29) + "BENCHMARK_ID".ljust(25) + "VERSIONS")

    def _format_versions(versions):
        return ", ".join(f"{d['version']}" for d in versions)

    print(
        "\n".join(
            f"- {info['name']}:".ljust(29)
            + f"{id_.ljust(25)}{_format_versions(info['versions'])}"
            for id_, info in benchmarks.items()
        )
    )


benchmark_cli.add_command(run)
benchmark_cli.add_command(list_benchmarks)
