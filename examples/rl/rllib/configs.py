import argparse
import multiprocessing
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union

try:
    from ray.rllib.algorithms.algorithm import AlgorithmConfig
    from ray.rllib.algorithms.callbacks import DefaultCallbacks
    from ray.rllib.algorithms.pg import PGConfig
    from ray.tune.search.sample import Integer as IntegerDomain
except Exception as e:
    from smarts.core.utils.custom_exceptions import RayException

    raise RayException.required_to("rllib.py")


def gen_pg_config(
    scenario,
    envision,
    rollout_fragment_length,
    train_batch_size,
    num_workers,
    log_level: Literal["DEBUG", "INFO", "WARN", "ERROR"],
    seed: Union[int, IntegerDomain],
    rllib_policies: Dict[str, Any],
    agent_specs: Dict[str, Any],
    callbacks: Optional[DefaultCallbacks],
) -> AlgorithmConfig:
    assert len(set(rllib_policies.keys()).difference(agent_specs)) == 0
    algo_config = (
        PGConfig()
        .environment(
            env="rllib_hiway-v0",
            env_config={
                "seed": seed,
                "scenarios": [str(Path(scenario).expanduser().resolve().absolute())],
                "headless": not envision,
                "agent_specs": agent_specs,
                "observation_options": "multi_agent",
            },
            disable_env_checking=True,
        )
        .framework(framework="tf2", eager_tracing=True)
        .rollouts(
            rollout_fragment_length=rollout_fragment_length,
            num_rollout_workers=num_workers,
            num_envs_per_worker=1,
            enable_tf1_exec_eagerly=True,
        )
        .training(
            lr_schedule=[(0, 1e-3), (1e3, 5e-4), (1e5, 1e-4), (1e7, 5e-5), (1e8, 1e-5)],
            train_batch_size=train_batch_size,
        )
        .multi_agent(
            policies=rllib_policies,
            policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: f"{agent_id}",
        )
        .callbacks(callbacks_class=callbacks)
        .debugging(log_level=log_level)
    )
    return algo_config


def gen_parser(
    prog: str, default_result_dir: str, default_save_model_path: str
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog)
    parser.add_argument(
        "--scenario",
        type=str,
        default=str(Path(__file__).resolve().parents[3] / "scenarios/sumo/loop"),
        help="Scenario to run (see scenarios/ for some samples you can use)",
    )
    parser.add_argument(
        "--envision",
        action="store_true",
        help="Run simulation with Envision display.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of times to sample from hyperparameter space",
    )
    parser.add_argument(
        "--rollout_fragment_length",
        type=str,
        default="auto",
        help="Episodes are divided into fragments of this many steps for each rollout. In this example this will be ensured to be `1=<rollout_fragment_length<=train_batch_size`",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=2000,
        help="The training batch size. This value must be > 0.",
    )
    parser.add_argument(
        "--time_total_s",
        type=int,
        default=1 * 60 * 60,  # 1 hour
        help="Total time in seconds to run the simulation for. This is a rough end time as it will be checked per training batch.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The base random seed to use, intended to be mixed with --num_samples",
    )
    parser.add_argument(
        "--num_agents", type=int, default=2, help="Number of agents (one per policy)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=(multiprocessing.cpu_count() // 2 + 1),
        help="Number of workers (defaults to use all system cores)",
    )
    parser.add_argument(
        "--resume_training",
        default=False,
        action="store_true",
        help="Resume an errored or 'ctrl+c' cancelled training. This does not extend a fully run original experiment.",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default=default_result_dir,
        help="Directory containing results",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="ERROR",
        help="Log level (DEBUG|INFO|WARN|ERROR)",
    )
    parser.add_argument(
        "--checkpoint_num", type=int, default=None, help="Checkpoint number"
    )
    parser.add_argument(
        "--checkpoint_freq", type=int, default=3, help="Checkpoint frequency"
    )

    parser.add_argument(
        "--save_model_path",
        type=str,
        default=default_save_model_path,
        help="Destination path of where to copy the model when training is over",
    )
    return parser
