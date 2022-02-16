# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
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
import multiprocessing
import os
import subprocess
import sys
from multiprocessing import Process, Semaphore
from pathlib import Path
from threading import Thread
from typing import Sequence

import click


@click.group(name="scenario")
def scenario_cli():
    pass


@scenario_cli.command(name="build", help="Generate a single scenario")
@click.option(
    "--clean",
    is_flag=True,
    default=False,
    help="Clean previously generated artifacts first",
)
@click.option(
    "--allow-offset-map",
    is_flag=True,
    default=False,
    help="Allows road network to be offset from the origin. If not specified, creates a new network file if necessary.",
)
@click.argument("scenario", type=click.Path(exists=True), metavar="<scenario>")
def build_scenario(clean: bool, allow_offset_map: bool, scenario: str):
    _build_single_scenario(clean, allow_offset_map, scenario)


def _build_single_scenario(clean: bool, allow_offset_map: bool, scenario: str):
    click.echo(f"build-scenario {scenario}")
    if clean:
        _clean(scenario)

    scenario_root = Path(scenario)
    scenario_root_str = str(scenario_root)

    scenario_py = scenario_root / "scenario.py"
    if scenario_py.exists():
        _install_requirements(scenario_root)
        subprocess.check_call([sys.executable, scenario_py])

    from smarts.core.scenario import Scenario

    traffic_histories = Scenario.discover_traffic_histories(scenario_root_str)
    shift_to_origin = not allow_offset_map or bool(traffic_histories)

    map_spec = Scenario.discover_map(scenario_root_str, shift_to_origin=shift_to_origin)
    road_map, _ = map_spec.builder_fn(map_spec)
    if not road_map:
        click.echo(
            "No reference to a RoadNetwork file was found in {}, or one could not be created. "
            "Please make sure the path passed is a valid Scenario with RoadNetwork file required "
            "(or a way to create one) for scenario building.".format(scenario_root_str)
        )
        return

    road_map.to_glb(os.path.join(scenario_root, "map.glb"))


def _build_single_scenario_proc(
    clean: bool, allow_offset_map: bool, scenario: str, semaphore: Semaphore
):
    semaphore.acquire()
    try:
        _build_single_scenario(clean, allow_offset_map, scenario)
    finally:
        semaphore.release()


def _install_requirements(scenario_root):
    import importlib.resources as pkg_resources

    requirements_txt = scenario_root / "requirements.txt"
    if requirements_txt.exists():
        import zoo.policies

        with pkg_resources.path(zoo.policies, "") as path:
            # Serve policies through the static file server, then kill after
            # we've installed scenario requirements
            pip_index_proc = subprocess.Popen(
                ["twistd", "-n", "web", "--path", path],
                # Hide output to keep display simple
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )

            pip_install_cmd = [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-r",
                str(requirements_txt),
            ]

            click.echo(
                f"Installing scenario dependencies via '{' '.join(pip_install_cmd)}'"
            )

            try:
                subprocess.check_call(pip_install_cmd, stdout=subprocess.DEVNULL)
            finally:
                pip_index_proc.terminate()
                pip_index_proc.wait()


def _is_scenario_folder_to_build(path: str) -> bool:
    if os.path.exists(os.path.join(path, "waymo.yaml")):
        # for now, don't try to build Waymo scenarios...
        return False
    if os.path.exists(os.path.join(path, "scenario.py")):
        return True
    from smarts.sstudio.types import MapSpec

    map_spec = MapSpec(path)
    road_map, _ = map_spec.builder_fn(map_spec)
    return road_map is not None


@scenario_cli.command(
    name="build-all",
    help="Generate all scenarios under the given directories",
)
@click.option(
    "--clean",
    is_flag=True,
    default=False,
    help="Clean previously generated artifacts first",
)
@click.option(
    "--allow-offset-maps",
    is_flag=True,
    default=False,
    help="Allows road networks (maps) to be offset from the origin. If not specified, creates creates a new network file if necessary.",
)
@click.argument("scenarios", nargs=-1, metavar="<scenarios>")
def build_all_scenarios(clean: bool, allow_offset_maps: bool, scenarios: str):
    if not scenarios:
        # nargs=-1 in combination with a default value is not supported
        # if scenarios is not given, set /scenarios as default
        scenarios = ["scenarios"]

    concurrency = max(1, multiprocessing.cpu_count() - 1)
    sema = Semaphore(concurrency)
    all_processes = []
    for scenarios_path in scenarios:
        for subdir, _, _ in os.walk(scenarios_path):
            if _is_scenario_folder_to_build(subdir):
                p = Path(subdir)
                scenario = f"{scenarios_path}/{p.relative_to(scenarios_path)}"
                proc = Process(
                    target=_build_single_scenario_proc,
                    args=(clean, allow_offset_maps, scenario, sema),
                )
                all_processes.append((scenario, proc))
                proc.start()

    for scenario_path, proc in all_processes:
        click.echo(f"Waiting on {scenario_path} ...")
        proc.join()


@scenario_cli.command(name="clean")
@click.argument("scenario", type=click.Path(exists=True), metavar="<scenario>")
def clean_scenario(scenario: str):
    _clean(scenario)


def _clean(scenario: str):
    to_be_removed = [
        "map.glb",
        "bubbles.pkl",
        "missions.pkl",
        "flamegraph-perf.log",
        "flamegraph.svg",
        "flamegraph.html",
        "*.rou.xml",
        "*.rou.alt.xml",
        "social_agents/*",
        "traffic/*",
        "history_mission.pkl",
        "*.shf",
        "*-AUTOGEN.net.xml",
    ]
    p = Path(scenario)
    for file_name in to_be_removed:
        for f in p.glob(file_name):
            # Remove file
            f.unlink()


@scenario_cli.command(name="replay")
@click.option("-d", "--directory", multiple=True)
@click.option("-t", "--timestep", default=0.01, help="Timestep in seconds")
@click.option("--endpoint", default="ws://localhost:8081")
def replay(directory: Sequence[str], timestep: float, endpoint: str):
    from envision.client import Client as Envision

    for path in directory:
        jsonl_paths = list(Path(path).glob("*.jsonl"))
        click.echo(
            f"Replaying {len(jsonl_paths)} record(s) at path={path} with "
            f"timestep={timestep}s"
        )

        with multiprocessing.pool.ThreadPool(len(jsonl_paths)) as pool:
            pool.starmap(
                Envision.read_and_send,
                [(jsonl, endpoint, timestep) for jsonl in jsonl_paths],
            )


scenario_cli.add_command(build_scenario)
scenario_cli.add_command(build_all_scenarios)
scenario_cli.add_command(clean_scenario)
scenario_cli.add_command(replay)
