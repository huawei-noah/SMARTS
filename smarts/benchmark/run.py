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
_MAX_EPISODE_STEPS = 1000

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


def _compute(scenario_dir, ep_per_scenario=5, max_episode_steps=_MAX_EPISODE_STEPS):
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
        update = results[scenario_name].update
        for _ in range(num_episode_steps[scenario_name]):
            with timeit("Benchmark", print, funcs=[update]):
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
    update: Callable[[float], None]
    mean: Callable[[], float]
    std: Callable[[], float]


@dataclass
class _Result:
    num_episodes: int
    num_steps: int
    mean: float
    std: float


def welford()->Tuple[Callable[[float],None],Callable[[],float],Callable[[],float]]:
    # Welford's online mean and std computation
    # Reference: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#On-line_algorithm
    # Reference: https://www.adamsmith.haus/python/answers/how-to-find-a-running-standard-deviation-in-python 

    import math
    n = 0 # steps
    M = 0 
    S = 0

    def update(val:float):
        nonlocal n, M, S
        n = n + 1
        newM = M + (val - M)/n
        newS = S + (val - M)*(val - newM)
        M = newM
        S = newS

    def mean()->float:
        return M

    def std()->float:      
        nonlocal n, M, S
        if n == 1:
            return 0

        std = math.sqrt(S/(n-1))
        return std

    return update, mean, std


def _get_funcs() -> _Funcs:
    update, mean, std = welford()
    return _Funcs(
        update=lambda x:update(1000/x), # Steps per sec. Units: step/s
        mean=mean,
        std=std,
    )


def _readable(func: _Funcs, num_episodes: int, num_steps: int)->_Result:
    return _Result(
        num_episodes=num_episodes,
        num_steps=num_steps,
        mean=func.mean(),
        std=func.std(),
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
