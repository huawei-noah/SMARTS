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
# FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import List, Sequence

import click


@click.group(
    name="scenario",
    help="Generate, replay or clean scenarios. See `scl scenario COMMAND --help` for further options.",
)
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
    "--seed",
    type=int,
    default=42,
    help="Set the base seed of the scenario.",
)
@click.argument("scenario", type=click.Path(exists=True), metavar="<scenario>")
def build(clean: bool, scenario: str, seed: int):
    from smarts.sstudio.scenario_construction import build_scenario

    assert seed == None or isinstance(seed, (int))

    build_scenario(scenario=scenario, clean=clean, seed=seed, log=click.echo)


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
    "--seed",
    type=int,
    default=42,
    help="Set the base seed of the scenarios.",
)
@click.argument("scenarios", nargs=-1, metavar="<scenarios>")
def build_all(clean: bool, scenarios: List[str], seed: int):
    from smarts.sstudio.scenario_construction import build_scenarios

    build_scenarios(scenarios=scenarios, clean=clean, seed=seed, log=click.echo)


@scenario_cli.command(
    name="clean", help="Remove previously generated scenario artifacts."
)
@click.argument("scenario", type=click.Path(exists=True), metavar="<scenario>")
def clean_scenario(scenario: str):
    from smarts.sstudio.scenario_construction import clean_scenario

    clean_scenario(scenario)


@scenario_cli.command(name="replay", help="Play saved Envision data files in Envision.")
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

        with ThreadPool(len(jsonl_paths)) as pool:
            pool.starmap(
                Envision.read_and_send,
                [(jsonl, endpoint, timestep) for jsonl in jsonl_paths],
            )


scenario_cli.add_command(build)
scenario_cli.add_command(build_all)
scenario_cli.add_command(clean_scenario)
scenario_cli.add_command(replay)
