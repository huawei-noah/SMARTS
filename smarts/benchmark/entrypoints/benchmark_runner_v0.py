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
import argparse
import logging
from typing import List, Tuple

import gymnasium as gym
import ray

from smarts.benchmark import auto_install
from smarts.benchmark.driving_smarts import load_config
from smarts.benchmark.driving_smarts.v0 import DEFAULT_CONFIG
from smarts.env.gymnasium.wrappers.episode_limit import EpisodeLimit
from smarts.env.gymnasium.wrappers.metrics import CompetitionMetrics, Score
from smarts.zoo import registry as agent_registry

LOG_WORKERS = False


@ray.remote(num_returns=1)
def _eval_worker(name, env_config, episodes, agent_config):
    import warnings

    warnings.filterwarnings("ignore")
    env = gym.make(
        env_config["env"],
        scenario=env_config["scenario"],
        **env_config["shared_params"],
        **agent_config["interface"],
    )
    env = EpisodeLimit(env, episodes)
    env = CompetitionMetrics(env)
    agent = agent_registry.make_agent(
        locator=agent_config["locator"],
        **agent_config["params"],
    )

    observation, info = env.reset()
    try:
        while not info.get("reached_episode_limit"):
            try:
                # action: [global x-coordinate, global y-coordinate]
                action = {
                    agent_id: agent.act(obs) for agent_id, obs in observation.items()
                }
                # assert env.action_space.contains(action)
            except Exception:
                logging.error("Policy robustness failed.")
                # env.mark_policy_failure()
                terminated, truncated = False, True
            else:
                observation, reward, terminated, truncated, info = env.step(action)
            if terminated["__all__"] or truncated["__all__"]:
                observation, info = env.reset()
    finally:
        score = env.score()
        env.close()
    return name, score


def task_iterator(env_args, benchmark_args, agent_args):
    ray.init(num_cpus=4, log_to_driver=LOG_WORKERS)
    try:
        max_queued_tasks = 20
        unfinished_refs = []
        for name, env_config in env_args.items():
            if len(unfinished_refs) >= max_queued_tasks:
                ready_refs, unfinished_refs = ray.wait(unfinished_refs, num_returns=1)
                for name, score in ray.get(ready_refs):
                    yield name, score
            print(f"Evaluating {env_config['scenario']}...")
            unfinished_refs.append(
                _eval_worker.remote(
                    name=name,
                    env_config=env_config,
                    episodes=benchmark_args["eval_episodes"],
                    agent_config=agent_args,
                )
            )
        for name, score in ray.get(unfinished_refs):
            yield name, score
    finally:
        ray.shutdown()


def benchmark(benchmark_args, agent_args):
    print(f"Starting `{benchmark_args['name']}` benchmark.")
    env_args = {}
    for scenario in benchmark_args["standard_env"]["scenarios"]:
        env_args[f"{scenario}"] = dict(
            env=benchmark_args["standard_env"]["env"],
            scenario=scenario,
            shared_params=benchmark_args["shared_env_params"],
        )
    ## TODO MTA bubble environments
    # for seed in config["bubble_env"]["scenarios"]:
    #     env_configs[f"bubble_env_{seed}"] = partial(
    #         env=config["bubble_env"]["env"],
    #         scenario=seed,
    #         shared_params=config["shared_env_params"],
    #     )
    # TODO MTA: naturalistic environments
    named_scores = []

    for name, score in task_iterator(
        env_args=env_args,
        benchmark_args=benchmark_args,
        agent_args=agent_args,
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


def benchmark_from_configs(benchmark_config, agent_config=None, log=False):
    global LOG_WORKERS
    benchmark_args = load_config(benchmark_config)
    agent_args = {}
    if agent_config:
        agent_args = load_config(agent_config)
    auto_install(benchmark_args)
    LOG_WORKERS = log
    benchmark(
        benchmark_args=benchmark_args["benchmark"],
        agent_args={**benchmark_args["agent"], **agent_args},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Driving SMARTS Competition")
    parser.add_argument(
        "--benchmark-config",
        help="The benchmark configuration file",
        default=DEFAULT_CONFIG,
        type=str,
    )
    parser.add_argument(
        "--log-workers",
        help="If the workers should log",
        default=False,
        type=bool,
    )
    args = parser.parse_args()

    benchmark_from_configs(args.config, log=args.log_workers)
