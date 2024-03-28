# MIT License
#
# Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.
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
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Generator, Tuple

import gymnasium as gym
import psutil
import ray

from smarts.benchmark.driving_smarts import load_config
from smarts.core import config
from smarts.core.utils.core_logging import suppress_output
from smarts.core.utils.import_utils import import_module_from_file
from smarts.env.gymnasium.wrappers.metric.formula import FormulaBase, Score
from smarts.env.gymnasium.wrappers.metric.metrics import Metrics
from smarts.env.gymnasium.wrappers.metric.types import Record
from smarts.zoo import registry as agent_registry

LOG_WORKERS = False
ERROR_TOLERANT = False


@ray.remote(num_returns=1)
def _eval_worker(name, env_config, episodes, agent_locator, error_tolerant=False):
    return _eval_worker_local(name, env_config, episodes, agent_locator, error_tolerant)


def _eval_worker_local(name, env_config, episodes, agent_locator, error_tolerant=False):
    import warnings

    warnings.filterwarnings("ignore")
    env = gym.make(
        env_config["env"],
        scenario=env_config["scenario"],
        agent_interface=agent_registry.make(locator=agent_locator).interface,
        **env_config["kwargs"],
    )
    env = Metrics(env, formula_path=env_config["metric_formula"])
    agents = {
        agent_id: agent_registry.make_agent(locator=agent_locator)[0]
        for agent_id in env.agent_ids
    }

    obs, info = env.reset()
    current_resets = 0
    try:
        while current_resets < episodes:
            try:
                action = {
                    agent_id: agents[agent_id].act(agent_obs)
                    for agent_id, agent_obs in obs.items()
                }
                # assert env.action_space.contains(action)
            except Exception:
                logging.error("Policy robustness failed.")
                # # TODO MTA: mark policy failures
                # env.mark_policy_failure()
                if not error_tolerant:
                    raise
                terminated, truncated = False, True
            else:
                obs, reward, terminated, truncated, info = env.step(action)
            if terminated["__all__"] or truncated["__all__"]:
                current_resets += 1
                obs, info = env.reset()
    finally:
        records = env.records()
        env.close()
    return name, records


def _parallel_task_iterator(env_args, benchmark_args, agent_locator, *args, **_):
    requested_cpus: int = config()(
        "ray",
        "num_cpus",
        cast=int,
    )
    num_gpus = config()(
        "ray",
        "num_gpus",
        cast=float,
    )
    num_cpus = (
        requested_cpus
        if requested_cpus is not None
        else max(
            0, min(len(os.sched_getaffinity(0)), psutil.cpu_count(logical=False) or 4)
        )
    )
    log_to_driver = config()(
        "ray",
        "log_to_driver",
        cast=bool,
    )

    if num_cpus == 0 and num_gpus == 0:
        print(
            f"Resource count `[benchmark] {num_cpus=}` and `[benchmark] {num_gpus=}` is collectively 0. "
            "Using the serial runner instead."
        )
        for o in _serial_task_iterator(env_args, benchmark_args, agent_locator):
            yield o
            return

    with suppress_output(stdout=True):
        ray.init(num_cpus=num_cpus, num_gpus=num_gpus, log_to_driver=log_to_driver)
    try:
        max_queued_tasks = num_cpus
        unfinished_refs = []
        for name, env_config in env_args.items():
            if len(unfinished_refs) >= max_queued_tasks:
                ready_refs, unfinished_refs = ray.wait(unfinished_refs, num_returns=1)
                for name, records in ray.get(ready_refs):
                    yield name, records
            print(f"\nEvaluating {name}...")
            unfinished_refs.append(
                _eval_worker.remote(
                    name=name,
                    env_config=env_config,
                    episodes=benchmark_args["eval_episodes"],
                    agent_locator=agent_locator,
                    error_tolerant=ERROR_TOLERANT,
                )
            )
        for name, records in ray.get(unfinished_refs):
            yield name, records
    finally:
        ray.shutdown()


def _serial_task_iterator(
    env_args, benchmark_args, agent_locator, *args, **_
) -> Generator[Tuple[Any, Any], Any, None]:
    for name, env_config in env_args.items():
        print(f"\nEvaluating {name}...")
        name, records = _eval_worker_local(
            name=name,
            env_config=env_config,
            episodes=benchmark_args["eval_episodes"],
            agent_locator=agent_locator,
            error_tolerant=ERROR_TOLERANT,
        )
        yield name, records


def benchmark(benchmark_args, agent_locator) -> Tuple[Dict, Dict]:
    """Runs the benchmark using the following:
    Args:
        benchmark_args(dict): Arguments configuring the benchmark.
        agent_locator(str): Locator string for the registered agent.
        debug_log(bool): Whether the benchmark should log to `stdout`.
    """
    print(f"\n\n<-- Starting `{benchmark_args['name']}` benchmark -->\n")
    message = benchmark_args.get("message")
    if message is not None:
        print(message)

    debug = benchmark_args.get("debug", {})
    iterator = _serial_task_iterator if debug.get("serial") else _parallel_task_iterator

    root_dir = Path(__file__).resolve().parents[3]
    metric_formula_default = (
        root_dir / "smarts" / "env" / "gymnasium" / "wrappers" / "metric" / "formula.py"
    )
    weighted_scores, agent_scores = {}, {}
    for env_name, env_config in benchmark_args["envs"].items():
        metric_formula = (
            root_dir / x
            if (x := env_config.get("metric_formula", None)) != None
            else metric_formula_default
        )

        env_args = {}
        for scenario in env_config["scenarios"]:
            kwargs = dict(benchmark_args.get("shared_env_kwargs", {}))
            kwargs.update(env_config.get("kwargs", {}))
            env_args[f"{env_name}-{scenario}"] = dict(
                env=env_config.get("loc") or env_config["locator"],
                scenario=str(root_dir / scenario),
                kwargs=kwargs,
                metric_formula=metric_formula,
            )

        records_cumulative: Dict[str, Dict[str, Record]] = {}
        for _, records in iterator(
            env_args=env_args,
            benchmark_args=benchmark_args,
            agent_locator=agent_locator,
        ):
            records_cumulative.update(records)

        weighted_score = _get_weighted_score(
            records=records_cumulative, metric_formula=metric_formula
        )
        weighted_scores[env_name] = weighted_score
        print("\n\nOverall Weighted Score:\n")
        print(json.dumps(weighted_score, indent=2))
        agent_score = _get_agent_score(
            records=records_cumulative, metric_formula=metric_formula
        )
        agent_scores[env_name] = agent_score
        print("\n\nIndividual Agent Score:\n")
        print(json.dumps(agent_score, indent=2))

    print("\n<-- Evaluation complete -->\n")
    return weighted_scores, agent_scores


def _get_weighted_score(
    records: Dict[str, Dict[str, Record]], metric_formula: Path
) -> Score:
    import_module_from_file("custom_formula", metric_formula)
    from custom_formula import Formula

    formula: FormulaBase = Formula()

    score = formula.score(records=records)
    return score


def _get_agent_score(
    records: Dict[str, Dict[str, Record]], metric_formula: Path
) -> Dict[str, Dict[str, Score]]:
    import_module_from_file("custom_formula", metric_formula)
    from custom_formula import costs_to_score

    from smarts.env.gymnasium.wrappers.metric.formula import agent_scores

    score = agent_scores(records=records, func=costs_to_score)
    return score


def benchmark_from_configs(benchmark_config, agent_locator, debug_log=False):
    """Runs a benchmark given the following.

    Args:
        benchmark_config (str): The file path to the benchmark configuration.
        agent_locator (str): Locator string for the registered agent.
        debug_log (bool): Deprecated. Whether the benchmark should log to `stdout`.
    """
    benchmark_args = load_config(benchmark_config)

    benchmark(
        benchmark_args=benchmark_args["benchmark"],
        agent_locator=agent_locator,
    )
