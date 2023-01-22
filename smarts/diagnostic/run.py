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

import logging
import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import time
from typing import Any, Callable, Dict, Sequence

import cpuinfo
import gym
import matplotlib.pyplot as plt
import psutil
from mdutils.mdutils import MdUtils

import smarts
from cli.studio import build_scenarios
from smarts.core.scenario import Scenario
from smarts.core.utils.math import welford

_SEED = 42
_MAX_REPLAY_EPISODE_STEPS = 100
_MAX_EPISODE_STEPS = 1000

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


def _compute(scenario_dir, ep_per_scenario=10, max_episode_steps=_MAX_EPISODE_STEPS):
    build_scenarios(
        clean=False,
        scenarios=scenario_dir,
        seed=_SEED,
    )
    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=scenario_dir,
        shuffle_scenarios=False,
        sim_name="Diagnostic",
        agent_specs={},
        headless=True,
        sumo_headless=True,
        seed=_SEED,
    )
    scenarios = Scenario.get_scenario_list(scenario_dir)
    num_episodes = ep_per_scenario * len(scenarios)
    num_episode_steps = {
        str(scenario): (
            _MAX_REPLAY_EPISODE_STEPS
            if Scenario.discover_traffic_histories(scenario)
            else max_episode_steps
        )
        for scenario in scenarios
    }
    results = {str(scenario): _get_funcs() for scenario in scenarios}

    for _ in range(num_episodes):
        env.reset()
        scenario_name = (env.scenario_log)["scenario_map"]
        update = results[scenario_name].update
        for _ in range(num_episode_steps[scenario_name]):
            start = time()
            env.step({})
            update((time() - start) * 1000)
    env.close()

    records = {}
    for k, v in results.items():
        parsed_name = k.split("diagnostic/")[1]
        records[parsed_name] = _readable(func=v)

    return records


@dataclass
class _Funcs:
    update: Callable[[float], None]
    mean: Callable[[], float]
    std: Callable[[], float]
    steps: Callable[[], int]


@dataclass
class _Result:
    steps: int
    mean: float
    std: float


def _get_funcs() -> _Funcs:
    update, mean, std, steps = welford()
    return _Funcs(
        update=lambda x: update(1000 / x),  # Steps per sec. Units: step/s
        mean=mean,
        std=std,
        steps=steps,
    )


def _readable(func: _Funcs) -> _Result:
    return _Result(
        steps=func.steps(),
        mean=func.mean(),
        std=func.std(),
    )


def git_revision_short_hash() -> str:
    """
    Returns Git commit short hash.

    Returns:
        str: Commit hash.
    """
    return (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode("ascii")
        .strip()
    )


def git_branch() -> str:
    """
    Returns Git branch name.

    Returns:
        str: Branch name.
    """
    return (
        subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        .decode("ascii")
        .strip()
    )


def _write_report(results: Dict[str, Any]):
    datetime_now = datetime.now()
    folder = datetime_now.strftime("%Y_%m_%d_%H_%M_%S")
    dir = Path(__file__).resolve().parent / f"reports" / folder
    dir.mkdir(parents=True, exist_ok=True)

    mdFile = MdUtils(file_name=str(dir / "Report"), title="Benchmark Report")
    mdFile.write(f"SMARTS version: {smarts.VERSION}\n\n")
    mdFile.write(f"Date & Time: {datetime_now.strftime('%d/%m/%Y %H:%M:%S')}\n\n")
    mdFile.write(f"Branch: {git_branch()}\n\n")
    mdFile.write(f"Commit: {git_revision_short_hash()}\n\n")
    mdFile.write(f"OS Version: {platform.platform()}\n\n")
    mdFile.write(
        f"Processor: {cpuinfo.get_cpu_info()['brand_raw']} x {cpuinfo.get_cpu_info()['count']}\n\n"
    )
    mdFile.write(
        f"RAM: {str(round(psutil.virtual_memory().total / (1024.0 **3)))+' GB'}\n\n"
    )

    means = []
    stds = []
    scenarios = []
    content = ["Scenario(s)", "Total Time Steps", "Mean (steps/sec)", "Std (steps/sec)"]
    for scenario, data in results.items():
        scenarios.append(scenario)
        means.append(data.mean)
        stds.append(data.std)
        content.extend(
            [
                f"{scenario}",
                f"{data.steps}",
                f"{data.mean:.2f}",
                f"{data.std:.2f}",
            ]
        )

    mdFile.new_header(level=2, title="Result", add_table_of_contents="n")
    mdFile.new_table(columns=4, rows=len(list(results.keys())) + 1, text=content)
    plt.plot(scenarios, means)
    plt.errorbar(scenarios, means, stds, marker="o", capsize=3)
    plt.xlabel("Scenario")
    plt.ylabel("Steps / Sec")
    plt.savefig(dir / "Fig1.png")
    mdFile.new_paragraph(
        "<figure>"
        f"\n<img src={dir/'Fig1.png'} alt='line chart' style='width:500px;'/>"
        "\n</figure>"
    )

    mdFile.create_md_file()


def main(scenarios: Sequence[str]):
    """Run diagnostic.

    Args:
        scenarios (Sequence[str]): Scenarios to be timed.
    """

    results = {}
    for scenario in scenarios:
        path = str(Path(__file__).resolve().parent / scenario)
        logger.info("Diagnosing: %s", path)
        results.update(_compute(scenario_dir=[path]))

    _write_report(results)
