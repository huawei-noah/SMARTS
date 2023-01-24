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
import logging
import os
from typing import List, Tuple

import gymnasium as gym
import psutil
import ray

from smarts.benchmark.driving_smarts import load_config
from smarts.benchmark.driving_smarts.v0 import DEFAULT_CONFIG
from smarts.core.utils.logging import suppress_output
from smarts.env.gymnasium.wrappers.metrics import Metrics, Score
from smarts.zoo import registry as agent_registry

LOG_WORKERS = False
ERROR_TOLERANT = False


@ray.remote(num_returns=1)
def _eval_worker(name, env_config, episodes, agent_config, error_tolerant=False):
    import warnings

    warnings.filterwarnings("ignore")
    env = gym.make(
        env_config["env"],
        scenario=env_config["scenario"],
        **env_config["kwargs"],
        **agent_config["interface"],
    )
    env = Metrics(env)
    agent = agent_registry.make_agent(
        locator=agent_config["locator"],
        **agent_config["kwargs"],
    )

    observation, info = env.reset()
    current_resets = 0
    try:
        while current_resets < episodes:
            try:
                action = {
                    agent_id: agent.act(obs) for agent_id, obs in observation.items()
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
                observation, reward, terminated, truncated, info = env.step(action)
            if terminated["__all__"] or truncated["__all__"]:
                current_resets += 1
                observation, info = env.reset()
    finally:
        score = env.score()
        env.close()
    return name, score


def _task_iterator(env_args, benchmark_args, agent_args, log_workers):
    num_cpus = max(1, min(len(os.sched_getaffinity(0)), psutil.cpu_count(False) or 4))

    with suppress_output(stdout=True):
        ray.init(num_cpus=num_cpus, log_to_driver=log_workers)
    try:
        max_queued_tasks = 20
        unfinished_refs = []
        for name, env_config in env_args.items():
            if len(unfinished_refs) >= max_queued_tasks:
                ready_refs, unfinished_refs = ray.wait(unfinished_refs, num_returns=1)
                for name, score in ray.get(ready_refs):
                    yield name, score
            print(f"Evaluating {name}...")
            unfinished_refs.append(
                _eval_worker.remote(
                    name=name,
                    env_config=env_config,
                    episodes=benchmark_args["eval_episodes"],
                    agent_config=agent_args,
                    error_tolerant=ERROR_TOLERANT,
                )
            )
        for name, score in ray.get(unfinished_refs):
            yield name, score
    finally:
        ray.shutdown()


def benchmark(benchmark_args, agent_args, log_workers=False):
    """Runs the benchmark using the following:
    Args:
        benchmark_args(dict): Arguments configuring the benchmark.
        agent_args(dict): Arguments configuring the agent running in the benchmark.
        debug_log(bool): Whether the benchmark should log to stdout.
    """
    print(f"Starting `{benchmark_args['name']}` benchmark.")
    message = benchmark_args.get("message")
    if message is not None:
        print(message)
    env_args = {}
    for env_name, env_config in benchmark_args["envs"].items():
        for scenario in env_config["scenarios"]:
            kwargs = dict(benchmark_args.get("shared_env_kwargs", {}))
            kwargs.update(env_config.get("kwargs", {}))
            env_args[f"{env_name}-{scenario}"] = dict(
                env=env_config["loc"],
                scenario=scenario,
                kwargs=kwargs,
            )
    named_scores = []

    for name, score in _task_iterator(
        env_args=env_args,
        benchmark_args=benchmark_args,
        agent_args=agent_args,
        log_workers=log_workers,
    ):
        named_scores.append((name, score))
        print(f"Scoring {name}...")

    def format_one_line_scores(named_scores: List[Tuple[str, Score]]):
        name_just = 30
        headers = "SCENARIO".ljust(name_just) + "SCORE"
        return (
            headers
            + "\n"
            + "\n".join(
                f"- {name}:".ljust(name_just) + f"{score}"
                for name, score in named_scores
            )
        )

    def format_scores_total(named_scores: List[Tuple[str, Score]], scenario_count):
        score_sum = Score(*[sum(f) for f in zip(*[score for _, score in named_scores])])
        return "\n".join(
            f"- {k}: {v/scenario_count}" for k, v in score_sum._asdict().items()
        )

    print("Evaluation complete...")
    print()
    print(format_one_line_scores(named_scores))
    print()
    print("`Driving SMARTS V0` averaged result:")
    print(format_scores_total(named_scores, len(env_args) or 1))


def benchmark_from_configs(benchmark_config, agent_config, debug_log=False):
    """Runs a benchmark given the following.

    Args:
        benchmark_config(file path): The file path to the benchmark configuration.
        agent_config(file path): The file path to the agent configuration.
        debug_log(bool): Whether the benchmark should log to stdout.
    """
    benchmark_args = load_config(benchmark_config)
    agent_args = {}
    if agent_config:
        agent_args = load_config(agent_config)

    assert agent_args, f"""
    Cannot resolve `{agent_config}`. This should be in a location that can be resolved by python
    in python's `sys.path`.  (e.g. `<path>/custom/__init__.py` or`<path>/custom.py`) 
    
    Please ensure agent configuration:
      - file exists
      - file path is correct
      - contains correct data

    The benchmark cannot continue."""

    benchmark(
        benchmark_args=benchmark_args["benchmark"],
        agent_args=agent_args["agent"],
        log_workers=debug_log,
    )
