# smarts/benchmark/driving_smarts_competition_benchmark.py
import argparse
import logging
from functools import partial

import gymnasium as gym
import ray

from smarts.benchmark import auto_install
from smarts.benchmark.driving_smarts import DEFAULT_CONFIG, load_config
from smarts.env.gymnasium.wrappers.episode_limit import EpisodeLimit
from smarts.env.gymnasium.wrappers.metrics import CompetitionMetrics
from smarts.zoo import registry as agent_registry


@ray.remote(num_returns=1)
def _eval_worker(name, env_config, episodes, agent_config):
    import warnings

    warnings.filterwarnings("ignore")
    env = gym.make(
        env_config["env"],
        scenario=env_config["scenario"],
        **env_config["shared_params"],
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
                terminated, truncated = True, True
            else:
                observation, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                observation, info = env.reset()
    finally:
        score = env.score()
        env.close()
    return name, score


def task_iterator(env_configs, benchmark_config):
    ray.init(num_cpus=4)
    try:
        max_queued_tasks = 20
        unfinished_refs = []
        for name, env_config in env_configs.items():
            if len(unfinished_refs) >= max_queued_tasks:
                ready_refs, unfinished_refs = ray.wait(unfinished_refs, num_returns=1)
                for name, score in ray.get(ready_refs):
                    # sum scores
                    yield name, score
            print(f"Evaluating {env_config['scenario']}...")
            unfinished_refs.append(
                _eval_worker.remote(
                    name=name,
                    env_config=env_config,
                    episodes=benchmark_config["eval_episodes"],
                    agent_config=benchmark_config["agent"],
                )
            )
        for name, score in ray.get(unfinished_refs):
            yield name, score
    finally:
        ray.shutdown()


def benchmark(benchmark_config):
    print(f"Starting `{benchmark_config['name']}` benchmark.")
    env_configs = {}
    for scenario in benchmark_config["standard_env"]["scenarios"]:
        env_configs[f"{scenario}"] = dict(
            env=benchmark_config["standard_env"]["env"],
            scenario=scenario,
            shared_params=benchmark_config["shared_env_params"],
        )
    ## TODO MTA bubble env
    # for seed in config["bubble_env"]["scenarios"]:
    #     env_configs[f"bubble_env_{seed}"] = partial(
    #         env=config["bubble_env"]["env"],
    #         scenario=seed,
    #         shared_params=config["shared_env_params"],
    #     )
    # TODO MTA: naturalistic
    total_score = {}

    def sum_scores(s, o):
        return {k: s.get(k, 0) + o.get(k, 0) for k in set(o)}

    for name, score in task_iterator(
        env_configs=env_configs, benchmark_config=benchmark_config
    ):
        total_score = sum_scores(total_score, score)
        print(f"Scoring {name}...")

    print("Evaluation complete...")
    print()
    print("`Driving SMARTS V0` result:")
    print("\n".join(f"- {k}: {v}" for k, v in total_score.items()))


if __name__ == "__main__":
    print(DEFAULT_CONFIG)
    parser = argparse.ArgumentParser("driving_smarts_competition")
    parser.add_argument(
        "--config",
        help="The benchmark configuration file",
        default=DEFAULT_CONFIG,
        type=str,
    )
    args = parser.parse_args()

    base_config = load_config(args.config)
    auto_install(base_config)
    benchmark(
        benchmark_config=base_config["benchmark"],
    )
