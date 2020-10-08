import argparse
import logging
import multiprocessing
import random
from pathlib import Path

import numpy as np
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining

import smarts
from smarts.core.utils.file import copy_tree
from smarts.env.rllib_hiway_env import RLlibHiWayEnv

from .rllib_agent import TrainingModel, rllib_agent

logging.basicConfig(level=logging.INFO)


# Add custom metrics to your tensorboard using these callbacks
# see: https://ray.readthedocs.io/en/latest/rllib-training.html#callbacks-and-custom-metrics
def on_episode_start(info):
    episode = info["episode"]
    episode.user_data["ego_speed"] = []


def on_episode_step(info):
    episode = info["episode"]
    single_agent_id = list(episode._agent_to_last_obs)[0]
    obs = episode.last_raw_obs_for(single_agent_id)
    episode.user_data["ego_speed"].append(obs["speed"])


def on_episode_end(info):
    episode = info["episode"]
    mean_ego_speed = np.mean(episode.user_data["ego_speed"])
    print(
        f"ep. {episode.episode_id:<12} ended;"
        f" length={episode.length:<6}"
        f" mean_ego_speed={mean_ego_speed:.2f}"
    )
    episode.custom_metrics["mean_ego_speed"] = mean_ego_speed


def explore(config):
    # ensure we collect enough timesteps to do sgd
    if config["train_batch_size"] < config["rollout_fragment_length"] * 2:
        config["train_batch_size"] = config["rollout_fragment_length"] * 2
    return config


def main(
    scenario,
    headless,
    time_total_s,
    seed,
    num_samples,
    num_agents,
    num_workers,
    resume_training,
    result_dir,
    checkpoint_num,
    save_model_path,
):
    pbt = PopulationBasedTraining(
        time_attr="time_total_s",
        metric="episode_reward_mean",
        mode="max",
        perturbation_interval=300,
        resample_probability=0.25,
        # Specifies the mutations of these hyperparams
        hyperparam_mutations={
            "lr": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
            "rollout_fragment_length": lambda: random.randint(128, 16384),
            "train_batch_size": lambda: random.randint(2000, 160000),
        },
        # Specifies additional mutations after hyperparam_mutations is applied
        custom_explore_fn=explore,
    )

    # XXX: There is a bug in Ray where we can only export a trained model if
    #      the policy it's attached to is named 'default_policy'.
    #      See: https://github.com/ray-project/ray/issues/5339
    rllib_policies = {
        "default_policy": (
            None,
            rllib_agent["observation_space"],
            rllib_agent["action_space"],
            {"model": {"custom_model": TrainingModel.NAME}},
        )
    }

    smarts.core.seed(seed)
    tune_config = {
        "env": RLlibHiWayEnv,
        "log_level": "WARN",
        "num_workers": num_workers,
        "env_config": {
            "seed": tune.sample_from(lambda spec: random.randint(0, 300)),
            "scenarios": [str(Path(scenario).expanduser().resolve().absolute())],
            "headless": headless,
            "agent_specs": {
                f"AGENT-{i}": rllib_agent["agent_spec"] for i in range(num_agents)
            },
        },
        "multiagent": {"policies": rllib_policies},
        "callbacks": {
            "on_episode_start": on_episode_start,
            "on_episode_step": on_episode_step,
            "on_episode_end": on_episode_end,
        },
    }

    experiment_name = "rllib_example_multi"
    result_dir = Path(result_dir).expanduser().resolve().absolute()
    if checkpoint_num:
        checkpoint = str(
            result_dir / f"checkpoint_{checkpoint_num}" / f"checkpoint-{checkpoint_num}"
        )
    else:
        checkpoint = None

    print(f"Checkpointing at {str(result_dir)}")
    analysis = tune.run(
        "PG",
        name=experiment_name,
        stop={"time_total_s": time_total_s},
        checkpoint_freq=1,
        checkpoint_at_end=True,
        local_dir=str(result_dir),
        resume=resume_training,
        restore=checkpoint,
        max_failures=3,
        num_samples=num_samples,
        export_formats=["model", "checkpoint"],
        config=tune_config,
        scheduler=pbt,
    )

    print(analysis.dataframe().head())

    best_logdir = Path(analysis.get_best_logdir("episode_reward_max"))
    model_path = best_logdir / "model"

    copy_tree(str(model_path), save_model_path, overwrite=True)
    print(f"Wrote model to: {save_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("rllib-example")
    parser.add_argument(
        "scenario",
        help="Scenario to run (see scenarios/ for some samples you can use)",
        type=str,
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run simulation in headless mode",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of times to sample from hyperparameter space",
    )
    parser.add_argument(
        "--time_total_s",
        type=int,
        default=1 * 60 * 60,  # 1 hour
        help="Total time in seconds to run the simulation for",
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
        default="~/ray_results",
        help="Directory containing results",
    )
    parser.add_argument(
        "--checkpoint_num", type=int, default=None, help="Checkpoint number"
    )

    save_model_path = str(Path(__file__).expanduser().resolve().parent / "model")
    parser.add_argument(
        "--save_model_path",
        type=int,
        default=save_model_path,
        help="Destination path of where to copy the model when training is over",
    )
    args = parser.parse_args()
    main(
        scenario=args.scenario,
        headless=args.headless,
        time_total_s=args.time_total_s,
        seed=args.seed,
        num_samples=args.num_samples,
        num_agents=args.num_agents,
        num_workers=args.num_workers,
        resume_training=args.resume_training,
        result_dir=args.result_dir,
        checkpoint_num=args.checkpoint_num,
        save_model_path=args.save_model_path,
    )
