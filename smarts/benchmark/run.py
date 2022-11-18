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
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Tuple

import gym

from cli.studio import build_scenarios
from smarts.core.scenario import Scenario
from smarts.core.utils.logging import timeit

_SEED = 42
_MAX_REPLAY_EPISODE_STEPS = 100

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


def _compute(scenario_dir, ep_per_scenario=5, max_episode_steps=1000):
    build_scenarios(
        allow_offset_maps=False,
        clean=False,
        scenarios=scenario_dir,
        seed=_SEED,
    )
    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=scenario_dir,
        shuffle_scenarios=False,
        sim_name="Benchmark",
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
        avg_compute = results[scenario_name].avg_compute
        std_store = results[scenario_name].std_store
        with timeit("Benchmark", print, funcs=[avg_compute, std_store]):
            for _ in range(num_episode_steps[scenario_name]):
                env.step({})

    env.close()

    records = {}
    for k, v in results.items():
        records[k] = _readable(
            func=v, num_episodes=num_episodes, num_steps=num_episode_steps[k]
        )

    return records


@dataclass
class _Funcs:
    avg_compute: Callable[[float], float]
    avg_get: Callable[[], float]
    std_store: Callable[[float], None]
    std_get: Callable[[], float]


@dataclass
class _Result:
    num_episodes: int
    num_steps: int
    avg: float
    std: float
    steps_per_sec: float


def _avg() -> Tuple[Callable[[float], float], Callable[[], float]]:
    ave = 0
    step = 0

    def compute(val):
        nonlocal ave, step
        ave, step = _running_ave(prev_ave=ave, prev_step=step, new_val=val)
        return ave

    def get():
        nonlocal ave
        return ave

    return compute, get


def _running_ave(prev_ave: float, prev_step: int, new_val: float) -> Tuple[float, int]:
    new_step = prev_step + 1
    new_ave = prev_ave + (new_val - prev_ave) / new_step
    return new_ave, new_step


def _std() -> Tuple[Callable[[float], None], Callable[[], float]]:
    values = []

    def store(val):
        nonlocal values
        values.append(val)
        return

    def get():
        nonlocal values
        import statistics

        return statistics.stdev(values)

    return store, get


def _get_funcs() -> _Funcs:
    avg_compute, avg_get = _avg()
    std_store, std_get = _std()
    return _Funcs(
        avg_compute=avg_compute,
        avg_get=avg_get,
        std_store=std_store,
        std_get=std_get,
    )


def _readable(func: _Funcs, num_episodes: int, num_steps: int):
    avg = func.avg_get()
    std = func.std_get()
    steps_per_sec = num_steps / (avg / 1000)  # Units: Steps per Second

    return _Result(
        num_episodes=num_episodes,
        num_steps=num_steps,
        avg=avg,
        std=std,
        steps_per_sec=steps_per_sec,
    )


def main(scenarios):
    results = {}
    for scenario in scenarios:
        path = Path(__file__).resolve().parent / scenario
        logger.info("Benchmarking: %s", path)
        results.update(_compute(scenario_dir=[path]))

    print("----------------------------------------------")
    # scenarios = list(map(lambda s:s.split("benchmark/")[1], scenarios))
    print(results)


if __name__ == "__main__":
    results = main(scenarios=("n_sumo_actors"))
