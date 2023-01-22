# MIT License
#
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
import importlib
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

BENCHMARK_LISTING_FILE = Path(__file__).parent.absolute() / "benchmark_listing.yaml"


def auto_install_requirements(benchmark_spec: Dict[str, Any]):
    """Install dependencies as specified by the configuration given."""
    # TODO MTA: add configuration to configuration file
    requirements: List[str] = benchmark_spec.get("requirements", [])
    if len(requirements) > 0:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                *requirements,
            ]
        )


def _benchmark_at_version(target, version):
    versions = target["versions"]
    if not version:
        return versions[-1]

    for benchmark_at_version in versions:
        if benchmark_at_version["version"] == version:
            return benchmark_at_version

    def _format_versions(versions):
        return ", ".join(f"{d['version']}" for d in versions)

    raise KeyError(
        f"Version `{version}` was not found. Try from versions: {_format_versions(versions)}"
    )


def _get_entrypoint(path, name):
    module = importlib.import_module(path)
    entrypoint = module.__getattribute__(name)
    return entrypoint


def run_benchmark(
    benchmark_name: str,
    benchmark_version: Optional[float],
    agent_config: Path,
    benchmark_listing: Path,
    debug_log: bool = False,
    auto_install: bool = False,
):
    """Runs a benchmark with the given configuration. Use `scl benchmark list` to see the available
    benchmarks.

    Args:
        benchmark_name(str): The name of the benchmark to run.
        benchmark_version(float|None): The version of the benchmark.
        agent_config(Path): An agent configuration file.
        benchmark_listing(Path): A configuration file that lists benchmark metadata and must list
            the target benchmark.
        debug_log: Debug to stdout.
    """
    from smarts.core.utils.resources import load_yaml_config_with_substitution

    listing_dict = load_yaml_config_with_substitution(benchmark_listing)

    benchmarks = listing_dict["benchmarks"]

    try:
        benchmark_group = benchmarks[benchmark_name]
    except KeyError as err:
        raise RuntimeError(
            f"`{benchmark_name}` not found in config `{BENCHMARK_LISTING_FILE}`."
        ) from err

    benchmark_spec = _benchmark_at_version(benchmark_group, benchmark_version)

    if auto_install:
        auto_install_requirements(benchmark_spec)

    module, _, name = benchmark_spec["entrypoint"].rpartition(".")
    entrypoint = _get_entrypoint(module, name)
    entrypoint(
        **benchmark_spec.get("params", {}),
        agent_config=str(agent_config),
        debug_log=debug_log,
    )


def list_benchmarks(benchmark_listing):
    """Lists details of the currently available benchmarks."""
    from smarts.core.utils.resources import load_yaml_config_with_substitution

    return load_yaml_config_with_substitution(Path(benchmark_listing))
