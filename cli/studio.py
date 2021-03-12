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
import subprocess
import sys
from pathlib import Path
from threading import Thread

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
@click.argument("scenario", type=click.Path(exists=True), metavar="<scenario>")
def build_scenario(clean, scenario):
    _build_single_scenario(clean, scenario)


def _build_single_scenario(clean, scenario):
    import importlib.resources as pkg_resources

    from smarts.sstudio.sumo2mesh import generate_glb_from_sumo_network

    click.echo(f"build-scenario {scenario}")
    if clean:
        _clean(scenario)

    scenario_root = Path(scenario)
    map_net = scenario_root / "map.net.xml"
    map_glb = scenario_root / "map.glb"
    generate_glb_from_sumo_network(str(map_net), str(map_glb))

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

    scenario_py = scenario_root / "scenario.py"
    if scenario_py.exists():
        subprocess.check_call([sys.executable, scenario_py])


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
@click.argument("scenarios", nargs=-1, metavar="<scenarios>")
def build_all_scenarios(clean, scenarios):
    if not scenarios:
        # nargs=-1 in combination with a default value is not supported
        # if scenarios is not given, set /scenarios as default
        scenarios = ["scenarios"]
    builder_threads = {}
    for scenarios_path in scenarios:
        path = Path(scenarios_path)
        for p in path.rglob("*.net.xml"):
            scenario = f"{scenarios_path}/{p.parent.relative_to(scenarios_path)}"
            builder_thread = Thread(
                target=_build_single_scenario, args=(clean, scenario)
            )
            builder_thread.start()
            builder_threads[p] = builder_thread

    for scenario_path, builder_thread in builder_threads.items():
        click.echo(f"Waiting on {scenario_path} ...")
        builder_thread.join()


@scenario_cli.command(name="clean")
@click.argument("scenario", type=click.Path(exists=True), metavar="<scenario>")
def clean_scenario(scenario):
    _clean(scenario)


def _clean(scenario):
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
def replay(directory, timestep, endpoint):
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
