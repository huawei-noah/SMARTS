import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Tuple

import gym

from cli.studio import build_scenarios
from smarts.core.scenario import Scenario
from smarts.core.utils.logging import timeit

_SEED = 42

logging.basicConfig(level=logging.INFO)


def compute(scenario_dir, ep_per_scenario=5, max_episode_steps=1000):
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
    results = {str(scenario): get_funcs() for scenario in scenarios}

    for _ in range(num_episodes):
        env.reset()
        scenario_name = (env.scenario_log)["scenario_map"]
        avg_compute = results[scenario_name].avg_compute
        std_store = results[scenario_name].std_store
        with timeit("Benchmark", print, funcs=[avg_compute, std_store]):
            for _ in range(max_episode_steps):
                env.step({})

    env.close()

    records = {}
    for k, v in results.items():
        records[k] = _readable_results(
            func=v, num_episodes=num_episodes, num_steps=max_episode_steps
        )

    return records


def _readable_results(func, num_episodes, num_steps):
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


def avg() -> Tuple[Callable[[float], float], Callable[[], float]]:
    ave = 0
    step = 0

    def compute(val):
        nonlocal ave, step
        ave, step = _running_ave(prev_ave=ave, prev_step=step, new_val=val)
        print(f"Average: {ave},")
        return ave

    def get():
        nonlocal ave
        return ave

    return compute, get


def _running_ave(prev_ave: float, prev_step: int, new_val: float) -> Tuple[float, int]:
    new_step = prev_step + 1
    new_ave = prev_ave + (new_val - prev_ave) / new_step
    return new_ave, new_step


def std() -> Tuple[Callable[[float], None], Callable[[], float]]:
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


def get_funcs():
    avg_compute, avg_get = avg()
    std_store, std_get = std()
    return _Funcs(
        avg_compute=avg_compute,
        avg_get=avg_get,
        std_store=std_store,
        std_get=std_get,
    )


def main(scenarios):
    results = {}
    for scenario in scenarios:
        path = Path(__file__).resolve().parent / scenario
        results.update(compute(scenario_dir=[path]))

    print("----------------------------------------------")
    # scenarios = list(map(lambda s:s.split("benchmark/")[1], scenarios))
    print(results)


if __name__ == "__main__":
    results = main(scenarios=("n_sumo_actors"))
