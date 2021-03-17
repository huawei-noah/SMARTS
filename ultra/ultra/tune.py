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
import json
import os

# Set environment to better support Ray
os.environ["MKL_NUM_THREADS"] = "1"
import argparse

import gym
import torch
import yaml

import ray
from ray import tune
from functools import partial
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter

from smarts.zoo.registry import make
from ultra.evaluate import evaluation_check
from ultra.utils.episode import episodes
from ultra.baselines.ppo.ppo.tune_params import config as ppo_config

num_gpus = 1 if torch.cuda.is_available() else 0


# @ray.remote(num_gpus=num_gpus / 2, max_calls=1)
# @ray.remote(num_gpus=num_gpus / 2)
def tune_train(
    config,
    scenario_info,
    num_episodes,
    policy_classes,
    max_episode_steps,
    eval_info,
    timestep_sec,
    headless,
    seed,
    log_dir,
    # metrics,
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
        dones = {"__all__": False}
        infos = None
        episode.reset()

        while not dones["__all__"]:
            # Break if any of the agent's step counts is 1000000 or greater.
            if any([episode.get_itr(agent_id) >= 1000000 for agent_id in agents]):
                finished = True
                break

            # # Perform the evaluation check.
            # evaluation_check(
            #     agents=agents,
            #     agent_ids=agent_ids,
            #     policy_classes=agent_classes,
            #     episode=episode,
            #     log_dir=log_dir,
            #     max_episode_steps=max_episode_steps,
            #     **eval_info,
            #     **env.info,
            # )

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
        episode.record_episode()
        episode.record_tensorboard()

        tune_value = sum(
            [
                episode.info[episode.active_tag][agent_id].data[metric]
                for agent_id in agent_ids
            ]
        ) / len(agent_ids)
        tune.report(**{metric: tune_value})
        # # TODO: Make general for multi-agent...
        # tune_values = {
        #     metric: episode.info[episode.active_tag][0].data[metric]
        #     for metric in metrics
        # }
        # tune.report(**tune_values)

        if finished:
            break

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("intersection-training")
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
        default="sac",
    )
    parser.add_argument(
        "--episodes", help="Number of training episodes", type=int, default=1000000
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
        "--headless", help="Run without envision", type=bool, default=False
    )
    parser.add_argument(
        "--eval-episodes", help="Number of evaluation episodes", type=int, default=200
    )
    parser.add_argument(
        "--eval-rate",
        help="Evaluation rate based on number of observations",
        type=int,
        default=10000,
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
        default="logs",
        type=str,
    )
    parser.add_argument(
        "--policy-ids",
        help="Name of each specified policy",
        default=None,
        type=str,
    )

    args = parser.parse_args()

    METRIC = "episode_reward"
    MODE = "max"
    BEST_PARAMS_DIRECTORY = "tune_results/"
    policy_classes = ["ultra.baselines.ppo:ppo-v0"]
    config = ppo_config

    ray.init()

    scheduler = ASHAScheduler(
        metric=METRIC,
        mode=MODE,
        max_t=10,
        grace_period=1,
        reduction_factor=2,
    )
    reporter = CLIReporter(metric_columns=[METRIC, "accuracy", "training_iteration"])

    result = tune.run(
        partial(
            tune_train,
            scenario_info=(args.task, args.level),
            num_episodes=int(args.episodes),
            max_episode_steps=int(args.max_episode_steps),
            eval_info={
                "eval_rate": int(args.eval_rate),
                "eval_episodes": int(args.eval_episodes),
            },
            timestep_sec=float(args.timestep),
            headless=args.headless,
            policy_classes=policy_classes,
            seed=args.seed,
            log_dir=args.log_dir,
            metric=METRIC,
        ),
        config=config,
        num_samples=100,
        scheduler=scheduler,
        progress_reporter=reporter,
        local_dir="tune/",
    )

    best_trial = result.get_best_trial(METRIC, MODE, "last")
    best_config = result.get_best_config(METRIC, MODE, "last")

    print("Best trial:", best_trial)
    print("Best config:", best_config)
    print("Saving the best config to {}.".format(BEST_PARAMS_DIRECTORY))

    if not os.path.exists(BEST_PARAMS_DIRECTORY):
        os.makedirs(BEST_PARAMS_DIRECTORY)
    with open(
        os.path.join(BEST_PARAMS_DIRECTORY, "best_params.yaml"), "w"
    ) as best_params_file:
        yaml.dump(
            best_config, best_params_file, default_flow_style=False, sort_keys=False
        )
