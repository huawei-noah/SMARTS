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

import click


@click.group(
    name="benchmark",
    help="Utilities for running integrated ML benchmarks. See `scl benchmark COMMAND --help` for further options.",
)
def benchmark_cli():
    pass


@click.command(
    "run",
    help="Run the given benchmark. Like `scl benchmark run driving_smarts==0.0 ./baselines/driving_smarts/agent_config.yaml",
)
@click.argument("benchmark_id", nargs=1, metavar="<benchmark_id>")
@click.argument("agent_config", nargs=1, metavar="<agent_config>")
@click.option(
    "--debug-log",
    is_flag=True,
    default=False,
    help="Log the benchmark.",
)
def run(benchmark_id: str, agent_config: str, debug_log: bool):
    from smarts.benchmark import BENCHMARK_LISTING_FILE, run_benchmark

    benchmark_id, _, benchmark_version = benchmark_id.partition("==")

    run_benchmark(
        benchmark_id,
        float(benchmark_version),
        agent_config,
        BENCHMARK_LISTING_FILE,
        debug_log=debug_log,
    )


@click.command("list", help="Show available benchmarks.")
def list_benchmarks():
    from smarts.benchmark import BENCHMARK_LISTING_FILE
    from smarts.benchmark import list_benchmarks as l_benchmarks

    benchmarks = l_benchmarks(BENCHMARK_LISTING_FILE)["benchmarks"]

    print("BENCHMARK_NAME".ljust(29) + "BENCHMARK_ID".ljust(25) + "VERSIONS")

    def _format_versions(versions):
        return ", ".join(f"vers.{d['version']}" for d in versions)

    print(
        "\n".join(
            f"- {info['name']}:".ljust(29)
            + f"{id_.ljust(25)}{_format_versions(info['versions'])}"
            for id_, info in benchmarks.items()
        )
    )


benchmark_cli.add_command(run)
benchmark_cli.add_command(list_benchmarks)