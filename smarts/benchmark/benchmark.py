import logging
from mdutils.mdutils import MdUtils
import subprocess
import os
import gym
import smarts
from pygit2 import Repository
from smarts.core.utils.episodes import episodes
from smarts.core.smarts import SMARTS
from smarts.core.scenario import Scenario
from smarts.core.local_traffic_provider import LocalTrafficProvider
from smarts.core.utils.logging import timeit
from typing import Tuple, NamedTuple
from pathlib import Path
from smarts.core.utils.episodes import EpisodeLog
from datetime import datetime

logging.basicConfig(level=logging.INFO)


def compute_benchmark(scenarios, num_episodes=2, max_episode_steps=None):
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
    results = {}
    for ind, scenario in enumerate(scenario_iterator):
        print("\n---------------------------------------------------------------")
        print(
            f"\nRunning scenario {scenario._root.partition('benchmark/')[2]} for {num_episodes} episode(s)...\n"
        )
        ave_compute, ave_get = ave()
        std_store, std_get = std()
        for epi in range(num_episodes):
            env.reset()
            with timeit("Benchmark", print, funcs=[ave_compute, std_store]):
                for _ in range(max_episode_steps):
                    env.step({})
        print(
            f"\n{scenario._root.partition('benchmark/')[2]}: Avg time_step/sec = {ave_get()}, Std = {std_get()}\n"
        )
        results[scenario._root.partition("benchmark/")[2]] = BenchmarkOutput(
            ave_time=ave_get(),
            max_episode_steps=max_episode_steps,
            num_episodes=num_episodes,
            std=std_get(),
        )
    return results


def ave():
    ave = 0
    step = 0

    def compute(val):
        nonlocal ave, step
        ave, step = _running_ave(prev_ave=ave, prev_step=step, new_val=val)
        print(f"Average: {1000000/ave},")
        return 1000000 / ave

    def get():
        nonlocal ave
        return 1000000 / ave

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


def get_git_revision_short_hash() -> str:
    return (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode("ascii")
        .strip()
    )


class BenchmarkOutput(NamedTuple):
    ave_time: float
    max_episode_steps: int
    num_episodes: int
    std: float


# Write report in .md format
def write_report(results):
    now = datetime.now()
    current_date_and_time = now.strftime("%d_%m_%Y_%H_%M_%S")
    report_file_name = f"benchmark_result_{current_date_and_time}"
    mdFile = MdUtils(file_name=report_file_name, title="Benchmark Report")
    mdFile.write(f"SMARTS version: {smarts.VERSION}\n\n")
    mdFile.write(f"Date & Time: {now.strftime('%d/%m/%Y %H:%M:%S')}\n\n")
    mdFile.write(f"Branch: {Repository('.').head.shorthand}\n\n")
    mdFile.write(f"Commit: {get_git_revision_short_hash()}\n\n")
    mdFile.new_header(level=2, title="Intention", add_table_of_contents="n")
    mdFile.write("- Track the performance of SMARTS simulation for each version\n")
    mdFile.write("- Test the effects of performance improvements/optimizations\n")
    mdFile.new_header(level=2, title="Setup", add_table_of_contents="n")
    mdFile.new_paragraph(
        "- Dump different numbers of actors with different type respectively into 10 secs on a proper map without visualization.\n"
        "    - n social agents: 1, 10, 20, 50\n"
        "    - n data replay actors: 1, 10, 20, 50, 200\n"
        "    - n sumo traffic actors: 1, 10, 20, 50, 200\n"
        "    - 10 agents to n data replay actors: 1, 10, 20, 50\n"
        "    - 10 agent to n roads: 1, 10, 20, 50\n"
    )
    mdFile.new_header(level=2, title="Result", add_table_of_contents="n")
    scenario_list = ', '.join(str(s) for s in list(results.keys()))
    mdFile.new_paragraph(
        "The setup of this report is the following:\n"
        f"- Scenario(s): {scenario_list}\n" 
        f"- Total time steps: {list(results.values())[0].max_episode_steps}\n"
        f"- Number of episodes: {list(results.values())[0].num_episodes}"
    )
    # Write a table
    # content = ["Scenario(s)", "Mean", "Std"]
    # Write rows
    # for i in range(len(results.keys())):
    # # print(list(results.keys()))
    #     content.extend(
    #         [
    #             "1",
    #             # f"{list(results.keys())[i]}",
    #             f"{round(list(results.values())[0].ave_time,1)}", 
    #             f"{round(list(results.values())[0].std,2)}"
    #             ]
    #     )
    # mdFile.new_line()
    # mdFile.new_table(columns=3, rows=range(len(results.keys())), text=content)
    # print(list(results.keys())[0].split("/", 1)[0])
    mdFile.create_md_file()
    return report_file_name


def main(scenarios):
    results = compute_benchmark(scenarios)
    report_file_name = write_report(results)
    # Move report to report directory
    os.makedirs(f"{smarts.__path__[0]}/benchmark/benchmark_results", exist_ok=True)
    subprocess.run(
        [
            "mv",
            f"{os.getcwd()}/{report_file_name}.md",
            f"{smarts.__path__[0]}/benchmark/benchmark_results/{report_file_name}.md",
        ]
    )


if __name__ == "__main__":
    main()
