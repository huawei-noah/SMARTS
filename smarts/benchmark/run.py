import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Tuple
import os
import platform, psutil
import pandas as pd
import smarts
import gym
import subprocess
import cpuinfo

from cli.studio import build_scenarios
from datetime import datetime
from mdutils.mdutils import MdUtils
from pygit2 import Repository
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
        parsed_name=k.split("benchmark/")[1]
        records[parsed_name] = _readable(
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


def get_git_revision_short_hash() -> str:
    return (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode("ascii")
        .strip()
    )



# Write report in .md format
def write_report(results):
    os.makedirs(f"{smarts.__path__[0]}/benchmark/benchmark_results", exist_ok=True)
    now = datetime.now()
    current_date_and_time = now.strftime("%d_%m_%Y_%H_%M_%S")
    report_file_name = f"benchmark_result_{current_date_and_time}"
    mdFile = MdUtils(file_name=report_file_name, title="Benchmark Report")
    mdFile.write(f"SMARTS version: {smarts.VERSION}\n\n")
    mdFile.write(f"Date & Time: {now.strftime('%d/%m/%Y %H:%M:%S')}\n\n")
    mdFile.write(f"Branch: {Repository('.').head.shorthand}\n\n")
    mdFile.write(f"Commit: {get_git_revision_short_hash()}\n\n")
    mdFile.write(f"OS Version: {platform.platform()}\n\n")
    mdFile.write(
        f"Processor & RAM: {cpuinfo.get_cpu_info()['brand_raw']} x {cpuinfo.get_cpu_info()['count']} & {str(round(psutil.virtual_memory().total / (1024.0 **3)))+' GB'}"
    )
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
    scenario_list = ", ".join(str(s) for s in list(results.keys()))
    mdFile.new_paragraph(
        "The setup of this report is the following:\n"
        f"- Scenario(s): {scenario_list}\n"
        f"- Number of episodes: {list(results.values())[0].num_episodes}"
    )
    scenarios_list = []
    means_list = []
    # Write a table
    content = ["Scenario(s)","Total Time Steps","Mean(time_step/sec)", "Std"]
    # Write rows
    for scenario_path, data in results.items():
        scenarios_list.append(scenario_path)
        means_list.append(data.steps_per_sec)
        content.extend([f"{scenario_path}", f"{data.num_steps}", f"{data.steps_per_sec}", f"{data.std}"])
    mdFile.new_line()
    mdFile.new_table(columns=4, rows=len(list(results.keys())) + 1, text=content)
    # Draw a graph
    print(scenarios_list)
    df = pd.DataFrame(
        {
            "means": means_list,
        },
        index=scenarios_list,
    )
    print(df)
    graph = df.plot(kind="line", use_index=True, y="means", legend=False, marker=".")
    graph.get_figure().savefig(
        f"{smarts.__path__[0]}/benchmark/benchmark_results/{report_file_name}"
    )
    mdFile.new_paragraph(
        "<figure>"
        f"\n<img src='{smarts.__path__[0]}/benchmark/benchmark_results/{report_file_name}.png' alt='line chart' style='width:500px;'/>"
        "\n<figcaption align = 'center'> Figure 1 </figcaption>"
        "\n</figure>"
    )
    mdFile.create_md_file()

    subprocess.run(
        [
            "mv",
            f"{os.getcwd()}/{report_file_name}.md",
            f"{smarts.__path__[0]}/benchmark/benchmark_results/{report_file_name}.md",
        ]
    )

def main(scenarios):
    results = {}
    for scenario in scenarios:
        path = Path(__file__).resolve().parent / scenario
        logger.info("Benchmarking: %s", path)
        results.update(_compute(scenario_dir=[path]))

    print("----------------------------------------------")
    # scenarios = list(map(lambda s:s.split("benchmark/")[1], scenarios))
    write_report(results)

if __name__ == "__main__":
    results = main(scenarios=("n_sumo_actors"))
