import logging
import warnings
from pathlib import Path
from pprint import pprint as print
from typing import Dict, List, Literal, Optional, Union

import numpy as np
from ray.tune.experiment import Trial

# ray[rllib] is not the part of main dependency of the SMARTS package. It needs to be installed separately
# as a part of the smarts[train] dependency using the command "pip install -e .[train]. The following try block checks
# whether ray[rllib] was installed by user and raises an Exception warning the user to install it if not so.
try:
    from ray import tune
    from ray.rllib.algorithms.algorithm import AlgorithmConfig
    from ray.rllib.algorithms.callbacks import DefaultCallbacks
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.env.base_env import BaseEnv
    from ray.rllib.evaluation.episode import Episode
    from ray.rllib.evaluation.episode_v2 import EpisodeV2
    from ray.rllib.evaluation.rollout_worker import RolloutWorker
    from ray.rllib.policy.policy import Policy
    from ray.rllib.utils.typing import PolicyID
    from ray.tune.callback import Callback
    from ray.tune.registry import register_env
    from ray.tune.schedulers import PopulationBasedTraining
except Exception as e:
    from smarts.core.utils.custom_exceptions import RayException

    raise RayException.required_to("rllib.py")

import smarts
from smarts.core.utils.file import copy_tree
from smarts.env.rllib_hiway_env import RLlibHiWayEnv
from smarts.sstudio.scenario_construction import build_scenarios

if __name__ == "__main__":
    from configs import gen_parser
    from rllib_agent import TrainingModel, rllib_agent
else:
    from .configs import gen_parser
    from .rllib_agent import TrainingModel, rllib_agent

logging.basicConfig(level=logging.INFO)
register_env("rllib_hiway-v0", RLlibHiWayEnv)

# Add custom metrics to your tensorboard using these callbacks
# See: https://docs.ray.io/en/latest/rllib/rllib-examples.html#training-workflows
class AlgorithmCallbacks(DefaultCallbacks):
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

        mean_ego_reward = np.mean(episode.user_data["ego_reward"])
        print(
            f"ep. {episode.episode_id:<12} ended;"
            f" length={episode.length:<6}"
            f" mean_ego_reward={mean_ego_reward:.2f}"
        )
        episode.custom_metrics["mean_ego_reward"] = mean_ego_reward


class ExperimentCallback(Callback):
    def on_trial_error(self, iteration: int, trials: List[Trial], trial: Trial, **info):
        failed_trial = trial
        print(f"Trial has failed after {failed_trial.num_failures} failures.")
        return super().on_trial_error(iteration, trials, trial, **info)


def explore(config):
    # ensure we collect enough timesteps to do sgd
    rollout_fragment_length = config["rollout_fragment_length"]
    if (
        not isinstance(rollout_fragment_length, str)
        and config["train_batch_size"] < rollout_fragment_length * 2
    ):
        config["train_batch_size"] = int(rollout_fragment_length * 2)
    return config


def main(
    scenarios,
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
    log_level: Literal["DEBUG", "INFO", "WARN", "ERROR"],
    save_model_path,
):
    assert train_batch_size > 0, f"{train_batch_size.__name__} cannot be less than 1."
    if isinstance(rollout_fragment_length, str) and rollout_fragment_length != "auto":
        rollout_fragment_length = int(rollout_fragment_length)
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
    agent_specs = {f"AGENT-{i}": rllib_agent["agent_spec"] for i in range(num_agents)}

    smarts.core.seed(seed)
    assert len(set(rllib_policies.keys()).difference(agent_specs)) == 0
    algo_config: AlgorithmConfig = (
        PPOConfig()
        .environment(
            env=RLlibHiWayEnv,
            env_config={
                "seed": seed,
                "scenarios": [
                    str(Path(scenario).expanduser().resolve().absolute())
                    for scenario in scenarios
                ],
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
        .callbacks(callbacks_class=AlgorithmCallbacks)
        .debugging(log_level=log_level)
    )

    experiment_name = "rllib_example_multi"
    result_dir = Path(result_dir).expanduser().resolve().absolute()
    experiment_dir = result_dir / experiment_name

    print(f"======= Checkpointing at {str(result_dir)} =======")
    # Note that PBT modifies the hyperparameters during the run. This perturbation can be applied
    # to nearly anything passed to `Tuner(param_space=<config>)`.
    pbt = PopulationBasedTraining(
        time_attr="time_total_s",
        metric="episode_reward_mean",
        mode="max",
        perturbation_interval=20,
        resample_probability=0.25,
        # Specifies the mutations of these hyperparams
        hyperparam_mutations={
            "lr": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
            "train_batch_size": lambda: train_batch_size,
        },
        # Specifies additional mutations after hyperparam_mutations is applied
        custom_explore_fn=explore,
    )

    from ray import air

    run_config = air.RunConfig(
        name=experiment_name,
        stop={"time_total_s": time_total_s},
        callbacks=[ExperimentCallback()],
        storage_path=str(result_dir),
        checkpoint_config=air.CheckpointConfig(
            num_to_keep=3,
            checkpoint_frequency=checkpoint_freq,
            checkpoint_at_end=True,
        ),
        failure_config=air.FailureConfig(
            max_failures=3,
            fail_fast=False,
        ),
    )
    tune_config = tune.TuneConfig(
        num_samples=num_samples,
        scheduler=pbt,
        max_concurrent_trials=4,
    )
    trainable = "PPO"
    if resume_training:
        tuner = tune.Tuner.restore(
            str(experiment_dir),
            trainable=trainable,
            param_space=algo_config,
        )
    else:
        tuner = tune.Tuner(
            trainable=trainable,
            param_space=algo_config,
            tune_config=tune_config,
            run_config=run_config,
        )

    results = tuner.fit()
    # Get the best result based on a particular metric.
    best_result = results.get_best_result(metric="episode_reward_mean", mode="max")

    # Get the best checkpoint corresponding to the best result.
    best_checkpoint = best_result.checkpoint
    best_logdir = Path(best_checkpoint.path)
    model_path = best_logdir

    copy_tree(str(model_path), save_model_path, overwrite=True)
    print(f"Wrote model to: {save_model_path}")


if __name__ == "__main__":
    default_result_dir = str(
        Path(__file__).resolve().parent / "results" / "tune_pg_results"
    )
    parser = gen_parser("rllib-example", default_result_dir)
    parser.add_argument(
        "--num_samples",
        type=int,
        default=2,
        help="Number of times to sample from hyperparameter space.",
    )
    parser.add_argument(
        "--save_model_path",
        type=str,
        default=str(Path(__file__).expanduser().resolve().parent / "model"),
        help="Destination path of where to copy the model when training is over",
    )
    parser.add_argument(
        "--rollout_fragment_length",
        type=str,
        default="auto",
        help="Episodes are divided into fragments of this many steps for each rollout. In this example this will be ensured to be `1=<rollout_fragment_length<=train_batch_size`",
    )
    args = parser.parse_args()
    if not args.scenarios:
        args.scenarios = [
            str(Path(__file__).absolute().parents[2] / "scenarios" / "sumo" / "loop"),
        ]
    if args.num_samples < 2:
        warnings.warn(
            f"It is recommended to specify number_samples to be at least 2 to make use of PBT trial cloning.",
            category=UserWarning,
        )

    build_scenarios(scenarios=args.scenarios, clean=False, seed=args.seed)

    main(
        scenarios=args.scenarios,
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
        log_level=args.log_level,
        save_model_path=args.save_model_path,
    )
