import time
import sys
import multiprocessing
import subprocess
from pathlib import Path
from threading import Thread
import importlib.resources as pkg_resources

import click
import sh

from envision.client import Client as Envision
from smarts.sstudio.sumo2mesh import generate_glb_from_sumo_network


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
    click.echo(f"build-scenario {scenario}")
    if clean:
        _clean(scenario)

    map_net = f"{scenario}/map.net.xml"
    map_glb = f"{scenario}/map.glb"
    generate_glb_from_sumo_network(map_net, map_glb)

    scenario_py = f"{scenario}/scenario.py"
    if Path(scenario_py).exists():
        import zoo.policies

        with pkg_resources.path(zoo.policies, "") as path:
            # Serve policies through the static file server, then kill after the
            # scenario has been created
            proc = subprocess.Popen(
                f"twistd -n web --path {path}",
                shell=True,
                # Hide output to keep display simple
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )
            try:
                subprocess.check_call([sys.executable, scenario_py])
            finally:
                proc.terminate()


@scenario_cli.command(
    name="build-all", help="Generate all scenarios under the given directories",
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
