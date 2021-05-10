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
import glob

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
from ultra.utils.common import str_to_bool
from ultra.evaluate import evaluation_check, collect_evaluations
from ultra.utils.common import agent_pool_value
from ultra.utils.episode import episodes
from ultra.utils.curriculum.coordinator import Coordinator
from ultra.utils.curriculum.curriculum_info import CurriculumInfo
from ultra.utils.curriculum.scenario_data_handler import ScenarioDataHandler
from ultra.utils.curriculum.dynamic_scenarios import DynamicScenarios

num_gpus = 1 if torch.cuda.is_available() else 0


def load_model(experiment_dir):
    # Load relevant agent metadata.
    with open(
        os.path.join(experiment_dir, "agent_metadata.pkl"), "rb"
    ) as metadata_file:
        agent_metadata = pickle.load(metadata_file)

    for key in agent_metadata.keys():
        assert key in ["agent_ids", "agent_classes", "agent_specs"]

    # Extract the agent IDs and policy classes from the metadata and given models.
    agent_ids = [agent_id for agent_id in agent_metadata["agent_ids"]]

    agent_classes = {
        agent_id: agent_metadata["agent_classes"][agent_id] for agent_id in agent_ids
    }

    agent_checkpoint_directories = {
        agent_id: sorted(
            glob.glob(os.path.join(experiment_dir, "models", agent_id, "*")),
            key=lambda x: int(x.split("/")[-1]),
        )
        for agent_id in agent_ids
    }

    length_dir = len(agent_checkpoint_directories[agent_ids[0]])
    if length_dir > 1:
        print(
            f"\nThere are {length_dir} models inside in the experiment dir. Only the latest model >>> {agent_checkpoint_directories[agent_ids[0]][length_dir-1]} <<< will be trained\n"
        )

    current_checkpoint_directory = agent_checkpoint_directories[agent_ids[0]][
        length_dir - 1
    ]

    # Create the agent specifications matched with their associated ID and corresponding
    # checkpoint directory
    agent_specs = {
        agent_id: make(
            locator=agent_classes[agent_id],
            checkpoint_dir=current_checkpoint_directory,
            experiment_dir=experiment_dir,
            agent_id=agent_id,
        )
        for agent_id in agent_ids
    }

    return agent_ids, agent_classes, agent_specs


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
    experiment_dir,
    save_model_only,
    curriculum_mode,
    curriculum_metadata,
    policy_ids=None,
):
    torch.set_num_threads(1)
    total_step = 0
    finished = False
    evaluation_task_ids = dict()

    if experiment_dir:
        agent_ids, agent_classes, agent_specs = load_model(experiment_dir)
    else:
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

    if curriculum_mode:
        CurriculumInfo.initialize(curriculum_metadata["curriculum_dir"])
        if CurriculumInfo.static_curriculum_toggle is True:
            print("\n------------ Static Curriculum MODE : Enabled ------------\n")
            static_coordinator, scenario_info, num_episodes = static_curriculum_setup(
                curriculum_metadata, num_episodes
            )
            dynamic_coordinator = None
        elif CurriculumInfo.dynamic_curriculum_toggle is True:
            print("\n------------ Dynamic Curriculum MODE : Enabled ------------\n")
            dynamic_coordinator, scenario_info = dynamic_curriculum_setup(
                curriculum_metadata
            )
            static_coordinator = None
    else:
        print("\n------------ Curriculum MODE : Disabled ------------\n")
        dynamic_coordinator = None
        static_coordinator = None

    scenario_data_handler = ScenarioDataHandler("Train")

    # Create the environment.
    env = gym.make(
        "ultra.env:ultra-v0",
        agent_specs=agent_specs,
        scenario_info=scenario_info,
        headless=headless,
        timestep_sec=timestep_sec,
        seed=seed,
        curriculum_mode=curriculum_mode,
    )

    average_reached_goal = 0.0
    total_reached_goal = 0.0
    eval_after_grade = False

    for episode in episodes(num_episodes, etag=etag, log_dir=log_dir):
        if curriculum_mode is True:
            if CurriculumInfo.static_curriculum_toggle:
                graduate = static_coordinator.graduate(
                    episode.index, average_reached_goal
                )
                if graduate == True:
                    observations, scenario = env.reset(
                        True, static_coordinator.get_train_grade()
                    )
                    average_reached_goal = 0.0
                    grade_size = static_coordinator.get_grade_size()
                    scenario_data_handler.display_grade_scenario_distribution(
                        grade_size, static_coordinator.get_train_grade()
                    )
                    scenario_data_handler.save_grade_density(grade_size)
                    static_coordinator.episode_per_grade = 0
                    static_coordinator.end_warmup = False
                else:
                    observations, scenario = env.reset()
                # print("static_coordinator.episode_per_grade:", static_coordinator.episode_per_grade)
            elif (
                CurriculumInfo.dynamic_curriculum_toggle is True
                and episode.index % CurriculumInfo.sampling_rate == 0
            ):
                observations, scenario = env.reset(
                    switch=True, grade=CurriculumInfo.tasks_levels_used
                )
        else:
            observations, scenario = env.reset()

        density_counter = scenario_data_handler.record_density_data(
            scenario["scenario_density"]
        )
        scenario["density_counter"] = density_counter

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

        if curriculum_mode is True:
            if (CurriculumInfo.static_curriculum_toggle == True) and (
                graduate == True and CurriculumInfo.eval_per_grade == True
            ):
                eval_after_grade = True
        evaluation_check(
            agents=agents,
            agent_ids=agent_ids,
            policy_classes=agent_classes,
            episode=episode,
            log_dir=log_dir,
            curriculum_metadata=curriculum_metadata,
            curriculum_mode=curriculum_mode,
            static_coordinator=static_coordinator,
            max_episode_steps=max_episode_steps,
            evaluation_task_ids=evaluation_task_ids,
            eval_after_grade=eval_after_grade,
            save_model_only=save_model_only,
            **eval_info,
            **env.info,
        )
        collect_evaluations(evaluation_task_ids=evaluation_task_ids)

        if curriculum_mode == True:
            eval_after_grade = False
            if CurriculumInfo.static_curriculum_toggle == True:
                if static_coordinator.check_cycle_condition(episode.index):
                    print("No cycling of grades -> run completed")
                    break

        while not dones["__all__"]:
            # Break if any of the agent's step counts is 10000000 or greater.
            if any([episode.get_itr(agent_id) >= 10000000 for agent_id in agents]):
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

        episode.record_scenario_info(agents, scenario)
        episode.record_episode()
        episode.record_tensorboard(recording_step=episode.index)

        if curriculum_mode is True:
            if CurriculumInfo.static_curriculum_toggle is True:
                (
                    average_reached_goal,
                    total_reached_goal,
                ) = calculate_average_reached_goal(
                    episode, total_reached_goal, agents, average_reached_goal
                )
                if (episode.index + 1) % CurriculumInfo.pass_based_sample_rate == 0:
                    print(f"({episode.index + 1}) ARG: {average_reached_goal}")
            elif CurriculumInfo.dynamic_curriculum_toggle is True:
                if (
                    episode.index + 1
                ) % CurriculumInfo.sampling_rate == 0:  # Set sample rate (flag needs to be set)
                    scenario_data_handler.display_grade_scenario_distribution(
                        CurriculumInfo.sampling_rate
                    )
                    scenario_data_handler.save_grade_density(
                        CurriculumInfo.sampling_rate
                    )
                    print("Changing distribution ...")
                    dynamic_coordinator.change_distribution()
                    print("Resetting scenario pool ...")
                    dynamic_coordinator.reset_scenario_pool(
                        CurriculumInfo.tasks_levels_used
                    )

        if finished:
            break

    # print(static_coordinator.get_checkpoints())

    if curriculum_mode is False:
        scenario_data_handler.display_grade_scenario_distribution(num_episodes)
        scenario_data_handler.save_grade_density(num_episodes)

    filepath = os.path.join(episode.experiment_dir, "Train.csv")
    scenario_data_handler.plot_densities_data(filepath, curriculum_mode)

    print(
        "\nTotal scenario occurances:", scenario_data_handler.overall_densities_counter
    )

    # Wait on the remaining evaluations to finish.
    while collect_evaluations(evaluation_task_ids):
        time.sleep(0.1)

    env.close()


def static_curriculum_setup(curriculum_metadata, num_episodes):
    static_coordinator = Coordinator(
        curriculum_metadata["curriculum_dir"], num_episodes
    )
    # To build all scenarios from all grades
    if curriculum_metadata["curriculum_build_scenarios"]:
        static_coordinator.build_all_scenarios(
            curriculum_metadata["curriculum_scenarios_root_dir"],
            curriculum_metadata["curriculum_scenarios_save_dir"],
        )

    print("Number of grades:", static_coordinator.get_num_grades())
    static_coordinator.next_train_grade()
    scenario_info = tuple(static_coordinator.get_train_grade())

    if num_episodes % static_coordinator.get_num_grades() == 0:
        num_episodes += 1
        print("New max episodes (due to end case):", num_episodes)

    print("Num of episodes:", num_episodes)
    return static_coordinator, scenario_info, num_episodes


def dynamic_curriculum_setup(curriculum_metadata):
    dynamic_coordinator = DynamicScenarios(
        curriculum_metadata["curriculum_scenarios_root_dir"],
        curriculum_metadata["curriculum_scenarios_save_dir"],
        rate=CurriculumInfo.sampling_rate,
    )
    dynamic_coordinator.reset_scenario_pool(CurriculumInfo.tasks_levels_used)
    scenario_info = CurriculumInfo.tasks_levels_used
    return dynamic_coordinator, scenario_info


def calculate_average_reached_goal(
    episode, total_reached_goal, agents, average_reached_goal
):
    sample_rate = CurriculumInfo.pass_based_sample_rate

    if (episode.index + 1) % sample_rate == 0:
        total_reached_goal += episode.info[episode.active_tag][
            list(agents.keys())[0]
        ].data["reached_goal"]
        average_reached_goal = total_reached_goal / sample_rate
        total_reached_goal = 0.0
        return average_reached_goal, total_reached_goal
    else:
        total_reached_goal += episode.info[episode.active_tag][
            list(agents.keys())[0]
        ].data["reached_goal"]

    return average_reached_goal, total_reached_goal


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
        default=200,
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
        help="The number of training episodes to wait before running the evaluation",
        type=int,
        default=200,
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
        "--experiment-dir",
        help="Path to the base dir of trained model",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--save-model-only",
        help="Model is saved at checkpoint, but no evaluation occurs",
        default="False",
        type=str_to_bool,
    )
    parser.add_argument(
        "--curriculum-mode",
        help="Toggle grade based mode",
        default="False",
        type=str_to_bool,
    )
    parser.add_argument(
        "--curriculum-dir",
        help="local path to grade based (GB) curriculum dir. Local path is path from ultra/",
        type=str,
        default="scenarios/curriculum/",
    )
    parser.add_argument(
        "--curriculum-build-scenarios",
        help="Build all scenarios from curriculum",
        default="False",
        type=str_to_bool,
    )
    parser.add_argument(
        "--curriculum-scenarios-root-dir",
        help="Root directory where gb tasks are stored",
        type=str,
        default="ultra/scenarios",
    )
    parser.add_argument(
        "--curriculum-scenarios-save-dir",
        help="Save the scenarios in specified directory",
        type=str,
        default="ultra/scenarios/taskgb/",
    )

    base_dir = os.path.dirname(__file__)
    pool_path = os.path.join(base_dir, "agent_pool.json")
    args = parser.parse_args()

    # Obtain the policy class strings for each specified policy.
    policy_classes = [
        agent_pool_value(agent_name, "policy_class")
        for agent_name in args.policy.split(",")
    ]

    # Obtain the policy class IDs from the arguments.
    policy_ids = args.policy_ids.split(",") if args.policy_ids else None

    ray.init()
    train(
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
        experiment_dir=args.experiment_dir,
        save_model_only=args.save_model_only,
        curriculum_mode=args.curriculum_mode,
        curriculum_metadata={
            "curriculum_dir": args.curriculum_dir,
            "curriculum_build_scenarios": args.curriculum_build_scenarios,
            "curriculum_scenarios_root_dir": args.curriculum_scenarios_root_dir,
            "curriculum_scenarios_save_dir": args.curriculum_scenarios_save_dir,
        },
        policy_ids=policy_ids,
    )
