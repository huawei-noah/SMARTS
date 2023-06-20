import logging
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
    from ray.rllib.algorithms.pg import PGConfig
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
from smarts.sstudio.scenario_construction import build_scenario

if __name__ == "__main__":
    from configs import gen_parser
    from rllib_agent import TrainingModel, rllib_agent
else:
    from .configs import gen_parser
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
        episode.user_data["ego_speed"] = []

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
            episode.user_data["ego_speed"].append(infos["speed"])

    @staticmethod
    def on_episode_end(
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: Union[Episode, EpisodeV2],
        env_index: int,
        **kwargs,
    ):

        mean_ego_speed = np.mean(episode.user_data["ego_speed"])
        print(
            f"ep. {episode.episode_id:<12} ended;"
            f" length={episode.length:<6}"
            f" mean_ego_speed={mean_ego_speed:.2f}"
        )
        episode.custom_metrics["mean_ego_speed"] = mean_ego_speed


class TrialCallback(Callback):
    def on_trial_error(self, iteration: int, trials: List[Trial], trial: Trial, **info):
        t = trials[-1]
        path = Path(t.local_path)
        with open(path.parent / "meta", "wt") as f:
            f.write(str(path))
        return super().on_trial_error(iteration, trials, trial, **info)

    def on_experiment_end(self, trials: List[Trial], **info):
        t = trials[-1]
        path = Path(t.local_path)
        with open(path.parent / "meta", "wt") as f:
            f.write(str(path))
            print(f"Saved to {f.name}")
        return super().on_experiment_end(trials, **info)


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
    log_level: Literal["DEBUG", "INFO", "WARN", "ERROR"],
    save_model_path,
):
    assert train_batch_size > 0, f"{train_batch_size.__name__} cannot be less than 1."
    if (
        isinstance(rollout_fragment_length, int)
        and rollout_fragment_length > train_batch_size
    ):
        rollout_fragment_length = train_batch_size

    agent_values = {
        "agent_specs": {
            f"AGENT-{i}": rllib_agent["agent_spec"] for i in range(num_agents)
        },
        "rllib_policies": {
            f"AGENT-{i}": (
                None,
                rllib_agent["observation_space"],
                rllib_agent["action_space"],
                {"model": {"custom_model": TrainingModel.NAME}},
            )
            for i in range(num_agents)
        },
    }
    rllib_policies = agent_values["rllib_policies"]
    agent_specs = agent_values["agent_specs"]

    smarts.core.seed(seed)
    assert len(set(rllib_policies.keys()).difference(agent_specs)) == 0
    algo_config: AlgorithmConfig = (
        PGConfig()
        .environment(
            env=RLlibHiWayEnv,
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
        .callbacks(callbacks_class=Callbacks)
        .debugging(log_level=log_level)
    )

    experiment_name = "rllib_example_multi"
    result_dir = Path(result_dir).expanduser().resolve().absolute()
    experiment_dir = result_dir / experiment_name

    print(f"======= Checkpointing at {str(result_dir)} =======")
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

    from ray import air

    run_config = air.RunConfig(
        name=experiment_name,
        stop={"time_total_s": time_total_s},
        callbacks=[TrialCallback()],
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
    trainable = "PG"
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

    best_logdir = Path(best_result.log_dir)
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
        default=1,
        help="Number of times to sample from hyperparameter space",
    )
    parser.add_argument(
        "--save_model_path",
        type=str,
        default=str(Path(__file__).expanduser().resolve().parent / "model"),
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
        log_level=args.log_level,
        save_model_path=args.save_model_path,
    )
