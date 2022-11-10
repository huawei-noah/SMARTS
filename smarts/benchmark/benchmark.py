import logging

import gym
import smarts
from smarts.core.utils.episodes import episodes
from smarts.core.smarts import SMARTS
from smarts.core.scenario import Scenario
from smarts.core.local_traffic_provider import LocalTrafficProvider
from smarts.core.utils.logging import timeit
from typing import Tuple
from pathlib import Path
from smarts.core.utils.episodes import EpisodeLog

logging.basicConfig(level=logging.INFO)


def main(scenarios, num_episodes=2, max_episode_steps=None):
    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=scenarios,
        agent_specs={},
        headless=True,
        sumo_headless=True,
    )
    if max_episode_steps is None:
        max_episode_steps = 1000

    scenario_iterator = Scenario.scenario_variations(
        scenarios, agents_to_be_briefed=[], circular=False, shuffle_scenarios=False
    )

    for ind, scenario in enumerate(scenario_iterator):
        print("\n---------------------------------------------------------------")
        print(
            f"\nRunning scenario {scenario._root.partition('k/')[2]} for {num_episodes} episode(s)...\n"
        )
        ave_compute, ave_get = ave()
        std_store, std_get = std()
        for epi in range(num_episodes):
            env.reset()
            with timeit("Benchmark", print, funcs=[ave_compute, std_store]):
                for _ in range(max_episode_steps):
                    env.step({})
        print(
            f"\n{scenario._root.partition('k/')[2]}: Avg time_step/sec = {ave_get()}, Std={std_get()}\n"
        )


def ave():
    ave = 0
    step = 0

    def compute(val):
        nonlocal ave, step
        ave, step = _running_ave(prev_ave=ave, prev_step=step, new_val=val)
        print(f"Average: {1000000/ave},")
        return 1000000/ave

    def get():
        nonlocal ave
        return 1000000/ave

    return compute, get


def _running_ave(prev_ave: float, prev_step: int, new_val: float) -> Tuple[float, int]:
    new_step = prev_step + 1
    new_ave = prev_ave + (new_val - prev_ave) / new_step
    return new_ave, new_step


def std():
    values = []

    def store(val):
        nonlocal values
        values.append(1000000 / val)
        print(values)
    def get():
        nonlocal values
        import statistics

        return statistics.stdev(values)

    return store, get
