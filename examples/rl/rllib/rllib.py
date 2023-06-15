import argparse
import logging
import multiprocessing
import random
from pathlib import Path
from pprint import pprint as print
from typing import Dict, Literal, Optional, Union

import numpy as np

# ray[rllib] is not the part of main dependency of the SMARTS package. It needs to be installed separately
# as a part of the smarts[train] dependency using the command "pip install -e .[train]. The following try block checks
# whether ray[rllib] was installed by user and raises an Exception warning the user to install it if not so.
try:
    from ray import tune
    from ray.rllib.algorithms.callbacks import DefaultCallbacks
    from ray.rllib.algorithms.pg import PGConfig
    from ray.rllib.env.base_env import BaseEnv
    from ray.rllib.evaluation.episode import Episode
    from ray.rllib.evaluation.episode_v2 import EpisodeV2
    from ray.rllib.evaluation.rollout_worker import RolloutWorker
    from ray.rllib.policy.policy import Policy
    from ray.rllib.utils.typing import PolicyID
    from ray.tune.registry import register_env
    from ray.tune.schedulers import PopulationBasedTraining
except Exception as e:
    raise
    from smarts.core.utils.custom_exceptions import RayException

    raise RayException.required_to("rllib.py")

import smarts
from smarts.core.utils.file import copy_tree
from smarts.env.rllib_hiway_env import RLlibHiWayEnv
from smarts.sstudio.scenario_construction import build_scenario

if __name__ == "__main__":
    from rllib_agent import TrainingModel, rllib_agent
else:
    from .rllib_agent import TrainingModel, rllib_agent

logging.basicConfig(level=logging.INFO)
register_env("rllib_hiway-v0", RLlibHiWayEnv)

# Add custom metrics to your tensorboard using these callbacks
# See: https://ray.readthedocs.io/en/latest/rllib-training.html#callbacks-and-custom-metrics
class Callbacks(DefaultCallbacks):
    @staticmethod
    def on_episode_start(
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: Union[Episode, EpisodeV2],
        env_index: int,
        **kwargs,
    ):

        episode.user_data["ego_reward"] = []

    @staticmethod
    def on_episode_step(
        worker: RolloutWorker,
        base_env: BaseEnv,
        episode: Union[Episode, EpisodeV2],
        env_index: int,
        **kwargs,
    ):
        single_agent_id = list(episode.get_agents())[0]
        infos = episode._last_infos.get(single_agent_id)
        if infos is not None:
            episode.user_data["ego_reward"].append(infos["reward"])

    @staticmethod
    def on_episode_end(
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: Union[Episode, EpisodeV2],
        env_index: int,
        **kwargs,
    ):

        mean_ego_speed = np.mean(episode.user_data["ego_reward"])
        print(
            f"ep. {episode.episode_id:<12} ended;"
            f" length={episode.length:<6}"
            f" mean_ego_reward={mean_ego_speed:.2f}"
        )
        episode.custom_metrics["mean_ego_reward"] = mean_ego_speed


def explore(config):
    # ensure we collect enough timesteps to do sgd
    if config["train_batch_size"] < config["rollout_fragment_length"] * 2:
        config["train_batch_size"] = config["rollout_fragment_length"] * 2
    return config


def main(
    scenario,
    envision,
    time_total_s,
    rollout_fragment_length,
    train_batch_size,
    seed,
    num_samples,
    num_agents,
    num_workers,
    resume_training,
    result_dir,
    checkpoint_freq: int,
    checkpoint_num: Optional[int],
    log_level: Literal["DEBUG", "INFO", "WARN", "ERROR"],
    save_model_path,
):
    assert train_batch_size > 0, f"{train_batch_size.__name__} cannot be less than 1."
    if (
        isinstance(rollout_fragment_length, int)
        and rollout_fragment_length > train_batch_size
    ):
        rollout_fragment_length = train_batch_size

    rllib_policies = {
        f"AGENT-{i}": (
            None,
            rllib_agent["observation_space"],
            rllib_agent["action_space"],
            {"model": {"custom_model": TrainingModel.NAME}},
        )
        for i in range(num_agents)
    }

    smarts.core.seed(seed)
    algo_config = (
        PGConfig()
        .environment(
            env="rllib_hiway-v0",
            env_config={
                "seed": 42,
                "scenarios": [str(Path(scenario).expanduser().resolve().absolute())],
                "headless": not envision,
                "agent_specs": {
                    f"AGENT-{i}": rllib_agent["agent_spec"] for i in range(num_agents)
                },
                "observation_options": "multi_agent",
            },
            disable_env_checking=True,
        )
        .framework(framework="tf2", eager_tracing=True)
        .rollouts(
            rollout_fragment_length=rollout_fragment_length,
            num_rollout_workers=num_workers,
            num_envs_per_worker=1,
        )
        .training(
            lr_schedule=[(0, 1e-3), (1e3, 5e-4), (1e5, 1e-4), (1e7, 5e-5), (1e8, 1e-5)],
            train_batch_size=train_batch_size,
        )
        .multi_agent(
            policies=rllib_policies,
            policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: f"{agent_id}",
        )
        # .callbacks(callbacks_class=Callbacks)
        .debugging(log_level=log_level)
    )

    experiment_name = "rllib_example_multi"
    result_dir = Path(result_dir).expanduser().resolve().absolute()

    def get_checkpoint_dir(num):
        checkpoint_dir = result_dir / f"checkpoint_{num}" / f"checkpoint-{num}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return checkpoint_dir

    if checkpoint_num:
        checkpoint = str(get_checkpoint_dir(checkpoint_num))
    else:
        checkpoint = None

    print(f"======= Checkpointing at {str(result_dir)} =======")

    ## Approach 1
    pbt = PopulationBasedTraining(
        time_attr="time_total_s",
        metric="episode_reward_mean",
        mode="max",
        perturbation_interval=300,
        resample_probability=0.25,
        # Specifies the mutations of these hyperparams
        # See: `ray.rllib.agents.trainer.COMMON_CONFIG` for common hyperparams
        hyperparam_mutations={
            "lr": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
            "rollout_fragment_length": lambda: rollout_fragment_length,
            "train_batch_size": lambda: train_batch_size,
        },
        # Specifies additional mutations after hyperparam_mutations is applied
        custom_explore_fn=explore,
    )
    # analysis = tune.run(
    #     "PG",
    #     name=experiment_name,
    #     stop={"time_total_s": time_total_s},
    #     checkpoint_freq=checkpoint_freq,
    #     checkpoint_at_end=True,
    #     local_dir=str(result_dir),
    #     resume=resume_training,
    #     restore=checkpoint,
    #     max_failures=3,
    #     num_samples=num_samples,
    #     export_formats=["model", "checkpoint"],
    #     config=algo_config,
    #     scheduler=pbt,
    # )

    # print(analysis.dataframe().head())

    # best_logdir = Path(analysis.get_best_logdir("episode_reward_max", mode="max"))
    # model_path = best_logdir / "model"

    # copy_tree(str(model_path), save_model_path, overwrite=True)
    # print(f"Wrote model to: {save_model_path}")

    ## Approach 2
    from ray.rllib.algorithms.algorithm import Algorithm

    algo = algo_config.build()
    if checkpoint is not None:
        Algorithm.load_checkpoint(algo, checkpoint=checkpoint)
    result = {}
    current_iteration = 0
    checkpoint_iteration = checkpoint_num or 0

    try:
        while result.get("time_total_s", 0) < time_total_s:
            result = algo.train()
            print(f"======== Iteration {result['training_iteration']} ========")
            print(result, depth=1)

            if current_iteration % checkpoint_freq == 0:
                checkpoint_dir = get_checkpoint_dir(checkpoint_iteration)
                print(f"======= Saving checkpoint {checkpoint_iteration} =======")
                algo.save_checkpoint(checkpoint_dir)
                checkpoint_iteration += 1
            current_iteration += 1
        algo.save_checkpoint(get_checkpoint_dir(checkpoint_iteration))
    finally:
        algo.save(get_checkpoint_dir("latest"))

    algo.stop()

    ## Approach 3
    # from ray import air
    # run_config = air.RunConfig(
    #     name=experiment_name,
    #     stop={"time_total_s": time_total_s},
    #     callbacks=[Callbacks],
    #     storage_path=result_dir,
    #     checkpoint_config=air.CheckpointConfig(
    #         num_to_keep=3,
    #         checkpoint_frequency=checkpoint_freq,
    #         checkpoint_at_end=True,
    #     ),
    #     failure_config=air.FailureConfig(
    #         max_failures=3,
    #         fail_fast=False,
    #     ),
    #     local_dir=str(result_dir),
    # )
    # tune_config = tune.TuneConfig(
    #     metric="episode_reward_mean",
    #     mode="max",
    #     num_samples=num_samples,
    #     scheduler=pbt,
    # )
    # tuner = tune.Tuner(
    #     "PPO",
    #     param_space=algo_config,
    #     tune_config=tune_config,
    #     run_config=run_config,
    # )

    # results = tuner.fit()
    # # Get the best result based on a particular metric.
    # best_result = results.get_best_result(metric="episode_reward_mean", mode="max")

    # # Get the best checkpoint corresponding to the best result.
    # best_checkpoint = best_result.checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser("rllib-example")
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
        help="Resume the last trained example",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "results"),
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

    save_model_path = str(Path(__file__).expanduser().resolve().parent / "model")
    parser.add_argument(
        "--save_model_path",
        type=str,
        default=save_model_path,
        help="Destination path of where to copy the model when training is over",
    )
    args = parser.parse_args()
    build_scenario(scenario=args.scenario, clean=False, seed=42)

    main(
        scenario=args.scenario,
        envision=args.envision,
        time_total_s=args.time_total_s,
        rollout_fragment_length=args.rollout_fragment_length,
        train_batch_size=args.train_batch_size,
        seed=args.seed,
        num_samples=args.num_samples,
        num_agents=args.num_agents,
        num_workers=args.num_workers,
        resume_training=args.resume_training,
        result_dir=args.result_dir,
        checkpoint_freq=max(args.checkpoint_freq, 1),
        checkpoint_num=args.checkpoint_num,
        log_level=args.log_level,
        save_model_path=args.save_model_path,
    )
