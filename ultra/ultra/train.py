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
import sys

from ultra.utils.ray import default_ray_kwargs

# Set environment to better support Ray
os.environ["MKL_NUM_THREADS"] = "1"
import argparse
import pickle
import time

import dill
import gym
import psutil
import ray
import torch
import matplotlib.pyplot as plt

from smarts.zoo.registry import make
from ultra.evaluate import evaluation_check
from ultra.utils.common import str_to_bool
from ultra.utils.episode import episodes
from ultra.utils.coordinator import Coordinator, CurriculumInfo, ScenarioDataHandler

num_gpus = 1 if torch.cuda.is_available() else 0


# @ray.remote(num_gpus=num_gpus / 2, max_calls=1)
@ray.remote(num_gpus=num_gpus / 2)
def train(
    scenario_info,
    num_episodes,
    policy_classes,
    max_episode_steps,
    eval_info,
    timestep_sec,
    headless,
    seed,
    log_dir,
    grade_mode,
    gb_info,
    policy_ids=None,
):
    torch.set_num_threads(1)
    total_step = 0
    finished = False

    # Make agent_ids in the form of 000, 001, ..., 010, 011, ..., 999, 1000, ...;
    # or use the provided policy_ids if available.
    agent_ids = (
        ["0" * max(0, 3 - len(str(i))) + str(i) for i in range(len(policy_classes))]
        if not policy_ids
        else policy_ids
    )
    # Ensure there is an ID for each policy, and a policy for each ID.
    assert len(agent_ids) == len(policy_classes), (
        "The number of agent IDs provided ({}) must be equal to "
        "the number of policy classes provided ({}).".format(
            len(agent_ids), len(policy_classes)
        )
    )

    # Assign the policy classes to their associated ID.
    agent_classes = {
        agent_id: policy_class
        for agent_id, policy_class in zip(agent_ids, policy_classes)
    }
    # Create the agent specifications matched with their associated ID.
    agent_specs = {
        agent_id: make(locator=policy_class, max_episode_steps=max_episode_steps)
        for agent_id, policy_class in agent_classes.items()
    }
    # Create the agents matched with their associated ID.
    agents = {
        agent_id: agent_spec.build_agent()
        for agent_id, agent_spec in agent_specs.items()
    }

    # Define an 'etag' for this experiment's data directory based off policy_classes.
    # E.g. From a ["ultra.baselines.dqn:dqn-v0", "ultra.baselines.ppo:ppo-v0"]
    # policy_classes list, transform it to an etag of "dqn-v0:ppo-v0".
    etag = ":".join([policy_class.split(":")[-1] for policy_class in policy_classes])

    if grade_mode:
        agent_coordinator, scenario_info = gb_setup(gb_info, num_episodes)
        scenario_data_handler = ScenarioDataHandler("Train")
    else:
        print("\n------------ GRADE MODE : Disabled ------------\n")
        agent_coordinator = None

    # Create the environment.
    env = gym.make(
        "ultra.env:ultra-v0",
        agent_specs=agent_specs,
        scenario_info=scenario_info,
        headless=headless,
        timestep_sec=timestep_sec,
        seed=seed,
        grade_mode=grade_mode,
    )

    old_episode = None

    average_scenarios_passed = 0.0
    total_scenarios_passed = 0.0

    old_episode = None
    for episode in episodes(num_episodes, etag=etag, log_dir=log_dir):
        if grade_mode:
            graduate = agent_coordinator.graduate(
                episode.index, average_scenarios_passed
            )
            if graduate == True:
                observations, scenario = env.reset(True, agent_coordinator.get_grade())
                average_scenarios_passed = 0.0
                grade_size = agent_coordinator.get_grade_size()
                scenario_data_handler.display_grade_scenario_distribution(grade_size)
                scenario_data_handler.save_grade_density(grade_size)
                agent_coordinator.episode_per_grade = 0
            else:
                observations, scenario = env.reset()

            if agent_coordinator.check_cycle_condition(episode.index):
                print("No cycling of grades -> run completed")
                break
            density_counter = scenario_data_handler.record_density_data(
                scenario["scenario_density"]
            )
            scenario["density_counter"] = density_counter
            # print("agent_coordinator.episode_per_grade:", agent_coordinator.episode_per_grade)
        else:
            # Reset the environment and retrieve the initial observations.
            observations = env.reset()
            scenario = None

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

        if eval_info["eval_episodes"] != 0:
            # Perform the evaluation check.
            evaluation_check(
                agents=agents,
                agent_ids=agent_ids,
                policy_classes=agent_classes,
                episode=episode,
                log_dir=log_dir,
                max_episode_steps=max_episode_steps,
                grade_mode=grade_mode,
                agent_coordinator=agent_coordinator,
                **eval_info,
                **env.info,
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

        episode.record_episode()
        episode.record_tensorboard()

        if grade_mode:
            episode.record_density_tensorboard(scenario)
            (
                average_scenarios_passed,
                total_scenarios_passed,
            ) = agent_coordinator.calculate_average_scenario_passed(
                episode, total_scenarios_passed, agents, average_scenarios_passed
            )

        if finished:
            break

    # print(agent_coordinator.get_checkpoints())

    if grade_mode:
        filepath = os.path.join(episode.experiment_dir, "Train.csv")
        scenario_data_handler.plot_densities_data(filepath)

    env.close()


def gb_setup(gb_info, num_episodes):
    agent_coordinator = Coordinator(gb_info["gb_curriculum_dir"], num_episodes)
    # To build all scenarios from all grades
    if gb_info["gb_build_scenarios"]:
        agent_coordinator.build_all_scenarios(
            gb_info["gb_scenarios_root_dir"], gb_info["gb_scenarios_save_dir"]
        )
    print(
        "\n------------ GRADE MODE : Enabled ------------\n Number of Intervals (grades):",
        agent_coordinator.get_num_of_grades(),
    )
    agent_coordinator.next_grade()
    scenario_info = tuple(agent_coordinator.get_grade())
    return agent_coordinator, scenario_info


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
        "--headless", help="Run without envision", type=str_to_bool, default="True"
    )
    parser.add_argument(
        "--eval-episodes", help="Number of evaluation episodes", type=int, default=200
    )
    parser.add_argument(
        "--eval-rate",
        help="Evaluation rate based on number of episodes",
        type=int,
        default=100,
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
    parser.add_argument(
        "--gb-mode",
        help="Toggle grade based mode",
        default=False,
        type=bool,
    )
    parser.add_argument(
        "--gb-curriculum-dir",
        help="local path to grade based (GB) curriculum dir. Local path is path from ultra/",
        type=str,
        default="../scenarios/grade_based_curriculum/",
    )
    parser.add_argument(
        "--gb-build-scenarios",
        help="Build all scenarios from curriculum",
        default=False,
        type=bool,
    )
    parser.add_argument(
        "--gb-scenarios-root-dir",
        help="Root directory where gb tasks are stored",
        type=str,
        default="ultra/scenarios",
    )
    parser.add_argument(
        "--gb-scenarios-save-dir",
        help="Save the scenarios in specified directory",
        type=str,
        default=None,
    )

    base_dir = os.path.dirname(__file__)
    pool_path = os.path.join(base_dir, "agent_pool.json")
    args = parser.parse_args()

    # Obtain the policy class strings for each specified policy.
    policy_classes = []
    with open(pool_path, "r") as f:
        data = json.load(f)
        for policy in args.policy.split(","):
            if policy in data["agents"].keys():
                policy_classes.append(
                    data["agents"][policy]["path"]
                    + ":"
                    + data["agents"][policy]["locator"]
                )
            else:
                raise ImportError("Invalid policy name. Please try again")

    # Obtain the policy class IDs from the arguments.
    policy_ids = args.policy_ids.split(",") if args.policy_ids else None

    ray.init()
    ray.wait(
        [
            train.remote(
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
                grade_mode=args.gb_mode,
                gb_info={
                    "gb_curriculum_dir": args.gb_curriculum_dir,
                    "gb_build_scenarios": args.gb_build_scenarios,
                    "gb_scenarios_root_dir": args.gb_scenarios_root_dir,
                    "gb_scenarios_save_dir": args.gb_scenarios_save_dir,
                },
                policy_ids=policy_ids,
            )
        ]
    )
