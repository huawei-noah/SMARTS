# MIT License
#
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
from functools import partial
import glob
import importlib
import os
import pickle
from typing import Sequence, Tuple

# Set environment to better support Ray
os.environ["MKL_NUM_THREADS"] = "1"
import argparse

import dill
import gym
import torch
import yaml

import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.analysis.experiment_analysis import ExperimentAnalysis
from ray.tune.schedulers import ASHAScheduler
from ray.tune.trial import Trial

from smarts.zoo.registry import make
from ultra.evaluate import evaluate_saved_models
from ultra.utils.common import agent_pool_value
from ultra.utils.episode import episodes

_AVAILABLE_TUNE_METRICS = [
    "episode_length",
    "episode_return",
    "env_score",
]
_AVAILABLE_TUNE_MODES = [
    "max",
    "min",
]
_AVAILABLE_TUNE_SCOPES = ["all", "last", "avg", "last-5-avg", "last-10-avg"]


# TODO: Replace tune_train with something like a 'rollout' function that
# can be used by both tune.py and train.py. This requires refactoring
# train.py so that it can...
# 1) Accept configs for the agents
# 2) Return/yield an Episode object (or information) about the episode
def tune_train(
    config,
    scenario_info,
    num_episodes,
    policy_classes,
    max_episode_steps,
    save_rate,
    timestep_sec,
    headless,
    seed,
    log_dir,
    metric,
):
    torch.set_num_threads(1)
    total_step = 0
    finished = False

    assert len(policy_classes) == 1, "Can only tune with single agent experiments."

    # Make agent_ids in the form of 000, 001, ..., 010, 011, ..., 999, 1000, ...
    agent_ids = [
        "0" * max(0, 3 - len(str(i))) + str(i) for i in range(len(policy_classes))
    ]
    # Assign the policy classes to their associated ID.
    agent_classes = {
        agent_id: policy_class
        for agent_id, policy_class in zip(agent_ids, policy_classes)
    }
    # Create the agent specifications matched with their associated ID.
    agent_specs = {
        agent_id: make(
            locator=policy_class,
            policy_params=config,
            max_episode_steps=max_episode_steps,
        )
        for agent_id, policy_class in agent_classes.items()
    }
    # Create the agents matched with their associated ID.
    agents = {
        agent_id: agent_spec.build_agent()
        for agent_id, agent_spec in agent_specs.items()
    }

    # Create the environment.
    env = gym.make(
        "ultra.env:ultra-v0",
        agent_specs=agent_specs,
        scenario_info=scenario_info,
        headless=headless,
        timestep_sec=timestep_sec,
        seed=seed,
    )

    # Define an 'etag' for this experiment's data directory based off policy_classes.
    # E.g. From a ["ultra.baselines.dqn:dqn-v0", "ultra.baselines.ppo:ppo-v0"]
    # policy_classes list, transform it to an etag of "dqn-v0:ppo-v0".
    etag = ":".join([policy_class.split(":")[-1] for policy_class in policy_classes])

    for episode in episodes(num_episodes, etag=etag, log_dir=log_dir):
        # Reset the environment and retrieve the initial observations.
        observations = env.reset()
        scenario_name = env.scenario_log["scenario_map"]
        dones = {"__all__": False}
        infos = None
        episode.reset()
        experiment_dir = episode.experiment_dir

        # Save relevant agent metadata.
        if not os.path.exists(f"{experiment_dir}/agent_metadata.pkl"):
            if not os.path.exists(experiment_dir):
                os.makedirs(experiment_dir)
            with open(f"{experiment_dir}/agent_metadata.pkl", "wb") as metadata_file:
                dill.dump(
                    {
                        "agent_ids": agent_ids,
                        "agent_classes": agent_classes,
                        "agent_specs": agent_specs,
                    },
                    metadata_file,
                    pickle.HIGHEST_PROTOCOL,
                )

        while not dones["__all__"]:
            # Break if any of the agent's step counts is 1000000 or greater.
            if any([episode.get_itr(agent_id) >= 1000000 for agent_id in agents]):
                finished = True
                break

            # Request and perform actions on each agent that received an observation.
            actions = {
                agent_id: agents[agent_id].act(observation, explore=True)
                for agent_id, observation in observations.items()
            }
            next_observations, rewards, dones, infos = env.step(actions)

            # Active agents are those that receive observations in this step and the next
            # step. Step each active agent (obtaining their network loss if applicable).
            active_agent_ids = observations.keys() & next_observations.keys()
            loss_outputs = {
                agent_id: agents[agent_id].step(
                    state=observations[agent_id],
                    action=actions[agent_id],
                    reward=rewards[agent_id],
                    next_state=next_observations[agent_id],
                    done=dones[agent_id],
                    info=infos[agent_id],
                )
                for agent_id in active_agent_ids
            }

            # Record the data from this episode.
            episode.record_step(
                agent_ids_to_record=active_agent_ids,
                infos=infos,
                rewards=rewards,
                total_step=total_step,
                loss_outputs=loss_outputs,
            )

            # Update variables for the next step.
            total_step += 1
            observations = next_observations

        # Normalize the data and record this episode on tensorboard.
        episode.record_episode(scenario_name)
        episode.record_tensorboard(recording_step=episode.index)

        # Save the agent if we have reached its save rate.
        if (episode.index + 1) % save_rate == 0:
            for agent_id in agent_ids:
                checkpoint_directory = episode.checkpoint_dir(agent_id, episode.index)
                agents[agent_id].save(checkpoint_directory)

        # Average the metric over the number of agents (1 agent).
        tune_value = sum(
            [
                episode.info[episode.active_tag][agent_id].data[metric]
                for agent_id in agent_ids
            ]
        ) / len(agent_ids)
        tune.report(**{metric: tune_value})

        if finished:
            break

    env.close()


def _save_best_params(best_conditions: Sequence[str], result: ExperimentAnalysis):
    best_log_dir = result.get_best_logdir(*best_conditions)
    best_config = result.get_best_config(*best_conditions)
    tune_experiment_dir = os.path.join(best_log_dir, "../")
    best_params_file_path = os.path.join(
        tune_experiment_dir, f"best_{args.metric}_params.yaml"
    )

    print(f"Saving best params saved to {best_params_file_path}.")
    with open(best_params_file_path, "w") as best_params_file:
        yaml.dump(
            best_config, best_params_file, default_flow_style=False, sort_keys=False
        )


def _perform_evaluation_on_best(
    best_conditions: Sequence[str],
    headless: bool,
    max_episode_steps: int,
    num_episodes: int,
    result: ExperimentAnalysis,
    scenario_info: Tuple[str, str],
    timestep: float,
):
    best_trial = result.get_best_trial(*best_conditions)
    best_log_dir = result.get_best_logdir(*best_conditions)
    evaluation_results_dir = os.path.join(best_log_dir, "../evaluation/")
    evaluation_experiment_dir = glob.glob(
        os.path.join(best_log_dir, args.log_dir, "*/")
    )[0]
    agents = os.listdir(os.path.join(evaluation_experiment_dir, "models"))

    print(f"Evaluating the best performing trial ({best_trial}).")
    evaluate_saved_models(
        experiment_dir=evaluation_experiment_dir,
        log_dir=evaluation_results_dir,
        headless=headless,
        max_episode_steps=max_episode_steps,
        agents=agents,
        num_episodes=num_episodes,
        scenario_info=scenario_info,
        timestep=timestep,
    )


def _default_trial_name_creator(trial: Trial):
    # trial.trainable_name is the string 'DEFAULT'
    # trial.trial_id is a string of numbers 'XXXXX_YYYYY'
    return f"{trial.trainable_name}_{trial.trial_id}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser("intersection-tuning")
    parser.add_argument(
        "--task", help="Tasks available : [0, 1, 2]", type=str, default="1"
    )
    parser.add_argument(
        "--level",
        help="Levels available : [easy, medium, hard, no-traffic]",
        type=str,
        default="easy",
    )
    parser.add_argument(
        "--policy",
        help="Policies available : [ppo, sac, td3, dqn, bdqn]",
        type=str,
        default="ppo",
    )
    parser.add_argument(
        "--episodes",
        help="Maximum number of tuning episodes for each sampled set of hyperparameters",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "--max-episode-steps",
        help="Maximum number of steps per episode",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "--timestep", help="Environment timestep (sec)", type=float, default=0.1
    )
    parser.add_argument(
        "--headless", help="Run without envision", type=bool, default=True
    )
    parser.add_argument(
        "--eval-episodes",
        help="Number of evaluation episodes to perform on the best config after tuning",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--save-rate",
        help="Save rate based on number of observations",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--seed",
        help="Environment seed",
        default=2,
        type=int,
    )
    parser.add_argument(
        "--log-dir",
        help="Log directory location",
        default="tune_logs/",
        type=str,
    )
    parser.add_argument(
        "--config-module",
        help="The module containing the tune config dictionary",
        default="ultra.baselines.ppo.ppo.tune_params",
        type=str,
    )
    parser.add_argument(
        "--metric",
        help="The value to optimize [{}]".format(", ".join(_AVAILABLE_TUNE_METRICS)),
        default="episode_reward",
        type=str,
    )
    parser.add_argument(
        "--mode",
        help="How to optimize the metric [{}]".format(", ".join(_AVAILABLE_TUNE_MODES)),
        default="max",
        type=str,
    )
    parser.add_argument(
        "--scope",
        help="How to compare the trials [{}]".format(", ".join(_AVAILABLE_TUNE_SCOPES)),
        default="last",
        type=str,
    )
    parser.add_argument(
        "--grace-period",
        help="Used by the scheduler to only stop trials at least this old in time",
        default=None,
        type=int,
    )
    parser.add_argument(
        "--reduction-factor",
        help="Used by the scheduler to set halving rate and amount",
        default=2,
        type=int,
    )
    parser.add_argument(
        "--brackets",
        help="Used by the scheduler as the number of brackets",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--num-samples",
        help="Number of samples to draw from the config to train",
        default=100,
        type=int,
    )

    args = parser.parse_args()

    assert (
        args.metric in _AVAILABLE_TUNE_METRICS
    ), f"Unsupported metric '{args.metric}'."
    assert args.mode in _AVAILABLE_TUNE_MODES, f"Unsupported mode '{args.mode}'."
    assert args.scope in _AVAILABLE_TUNE_SCOPES, f"Unsupported scope '{args.scope}'."

    # Obtain the policy class strings for each specified policy.
    policy_classes = [
        agent_pool_value(agent_name, "policy_class")
        for agent_name in args.policy.split(",")
    ]
    assert len(policy_classes) == 1, "Only single agent tuning is supported."

    config_module = importlib.import_module(args.config_module)
    config = importlib.import_module(args.config_module).config
    grace_period = (
        int(args.grace_period) if args.grace_period else int(args.episodes / 10)
    )

    ray.init()

    # If time_attr is "training_iteration", it increments for
    # each trial every time tune.report is called.
    scheduler = ASHAScheduler(
        time_attr="training_iteration",  # Used for comparing time.
        metric=args.metric,  # The training result objective value attribute.
        mode=args.mode,  # Whether to minimize or maximize the metric.
        max_t=int(args.episodes),  # Maximum time units (time_attr) per trial.
        grace_period=grace_period,  # Only stop trials at least this old in time.
        reduction_factor=int(
            args.reduction_factor
        ),  # Used to set halving rate and amount.
        brackets=int(args.brackets),  # Number of brackets.
    )
    reporter = CLIReporter(metric_columns=[args.metric, "training_iteration"])

    result = tune.run(
        partial(
            tune_train,
            scenario_info=(args.task, args.level),
            num_episodes=int(args.episodes),
            max_episode_steps=int(args.max_episode_steps),
            save_rate=int(args.save_rate),
            timestep_sec=float(args.timestep),
            headless=args.headless,
            policy_classes=policy_classes,
            seed=args.seed,
            log_dir=args.log_dir,
            metric=args.metric,
        ),
        config=config,
        num_samples=int(args.num_samples),
        scheduler=scheduler,
        progress_reporter=reporter,
        local_dir=args.log_dir,
        trial_name_creator=_default_trial_name_creator,
    )

    ray.shutdown()

    _save_best_params(
        best_conditions=[args.metric, args.mode, args.scope],
        result=result,
    )
    _perform_evaluation_on_best(
        best_conditions=[args.metric, args.mode, args.scope],
        headless=args.headless,
        max_episode_steps=int(args.max_episode_steps),
        num_episodes=int(args.eval_episodes),
        result=result,
        scenario_info=(args.task, args.level),
        timestep=float(args.timestep),
    )
