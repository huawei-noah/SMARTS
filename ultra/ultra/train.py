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
import copy
import json
import os
import sys
from typing import Any, Dict, Tuple

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

from smarts.zoo.registry import make
from ultra.evaluate import evaluation_check, collect_evaluations
from ultra.utils.common import (
    AgentSpecPlaceholder,
    agent_pool_value,
    gen_default_agent_id,
    gen_etag_from_locators,
    replace_placeholder,
)
from ultra.utils.episode import episodes

num_gpus = 1 if torch.cuda.is_available() else 0


def train(
    scenario_info: Tuple[str, str],
    num_episodes: int,
    agent_infos: Dict[str, Dict[str, Any]],
    max_episode_steps: int,
    eval_info: Dict[str, Any],
    timestep_sec: float,
    headless: bool,
    seed: int,
    log_dir: str,
):
    torch.set_num_threads(1)
    total_step = 0
    finished = False
    evaluation_task_ids = dict()

    agent_infos_copy = copy.deepcopy(agent_infos)

    # TODO: Replace explore placeholder...
    print("train agent_infos:", agent_infos)
    print("train agent_infos_copy:", agent_infos_copy)
    agent_specs = {
        agent_id: make(locator=agent_info["locator"], **agent_info["spec_train_params"])
        for agent_id, agent_info in agent_infos_copy.items()
    }
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

    etag = gen_etag_from_locators(
        [agent_info["locator"] for agent_info in agent_infos.values()]
    )

    old_episode = None
    for episode in episodes(num_episodes, etag=etag, log_dir=log_dir):
        # Reset the environment and retrieve the initial observations.
        observations = env.reset()
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
                        # TODO: Maybe keep the agent_ids and agent_classes for backwards
                        #       compatibility?
                        # "agent_ids": list(agent_locators.keys()),
                        # "agent_classes": agent_locators,
                        "agent_infos": agent_infos,
                        "agent_specs": agent_specs,
                    },
                    metadata_file,
                    pickle.HIGHEST_PROTOCOL,
                )

        evaluation_check(
            agents=agents,
            agent_infos=agent_infos,
            episode=episode,
            log_dir=log_dir,
            max_episode_steps=max_episode_steps,
            evaluation_task_ids=evaluation_task_ids,
            **eval_info,
            **env.info,
        )

        collect_evaluations(evaluation_task_ids=evaluation_task_ids)

        while not dones["__all__"]:
            # Break if any of the agent's step counts is 1000000 or greater.
            if any([episode.get_itr(agent_id) >= 1000000 for agent_id in agents]):
                finished = True
                break
            # Request and perform actions on each agent that received an observation.
            actions = {}
            for agent_id, observation in observations.items():
                try:
                    actions[agent_id] = agents[agent_id].act(observation, explore=True)
                except TypeError:
                    actions[agent_id] = agents[agent_id].act(observation)
            next_observations, rewards, dones, infos = env.step(actions)

            # Active agents are those that receive observations in this step and the next
            # step. Step each active agent (obtaining their network loss if applicable).
            active_agent_ids = observations.keys() & next_observations.keys()
            loss_outputs = {}
            for agent_id in active_agent_ids:
                try:
                    loss_outputs[agent_id] = agents[agent_id].step(
                        state=observations[agent_id],
                        action=actions[agent_id],
                        reward=rewards[agent_id],
                        next_state=next_observations[agent_id],
                        done=dones[agent_id],
                        info=infos[agent_id],
                    )
                except AttributeError:
                    loss_outputs[agent_id] = {}

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

        if finished:
            break

    # Wait on the remaining evaluations to finish.
    while collect_evaluations(evaluation_task_ids):
        time.sleep(0.1)

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
        default=200,
    )
    parser.add_argument(
        "--timestep", help="Environment timestep (sec)", type=float, default=0.1
    )
    parser.add_argument(
        "--headless",
        help="Run without envision",
        action="store_true",
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
    args = parser.parse_args()

    # Obtain the policy class strings for each specified policy.
    agent_locators = {
        gen_default_agent_id(agent_number): agent_pool_value(agent_name, "policy_class")
        for agent_number, agent_name in enumerate(args.policy.split(","))
    }

    if not all("ultra.baselines" in locator for locator in agent_locators.values()):
        raise "Cannot train non-baseline agents from the command line."

    agent_infos = {
        agent_id: {
            "locator": agent_locator,
            "spec_train_params": {
                "max_episode_steps": args.max_episode_steps,
            },
            "spec_eval_params": {
                "max_episode_steps": args.max_episode_steps,
                "checkpoint_dir": AgentSpecPlaceholder.CheckpointDirectory,
                "experiment_dir": AgentSpecPlaceholder.ExperimentDirectory,
                "agent_id": agent_id,
            },
        }
        for agent_id, agent_locator in agent_locators.items()
    }

    ray.init()
    train(
        scenario_info=(args.task, args.level),
        num_episodes=int(args.episodes),
        agent_infos=agent_infos,
        # agent_infos={
        #     "000": {
        #         "locator": "ultra.baselines.bdqn:bdqn-v0",
        #         "spec_train_params": {
        #             "max_episode_steps": 300,
        #         },
        #         "spec_eval_params": {
        #             "max_episode_steps": 300,
        #             "checkpoint_dir": AgentSpecPlaceholder.CheckpointDirectory,
        #             "experiment_dir": AgentSpecPlaceholder.ExperimentDirectory,
        #             "agent_id": "000",
        #         },
        #     },
        #     "000": {
        #         "locator": "open_agent:open_agent-v0",
        #         "spec_train_params": {
        #             "gains": {
        #                 "theta": 3.0,
        #                 "position": 4.0,
        #                 "obstacle": 3.0,
        #                 "u_accel": 0.1,
        #                 "u_yaw_rate": 1.0,
        #                 "terminal": 0.01,
        #                 "impatience": 0.01,
        #                 "speed": 0.01,
        #                 "rate": 1,
        #             },
        #             "debug": False,
        #             "aggressiveness": 1,
        #             "max_episode_steps": 300,
        #         },
        #         "spec_eval_params": {
        #             "gains": {
        #                 "theta": 3.0,
        #                 "position": 4.0,
        #                 "obstacle": 3.0,
        #                 "u_accel": 0.1,
        #                 "u_yaw_rate": 1.0,
        #                 "terminal": 0.01,
        #                 "impatience": 0.01,
        #                 "speed": 0.01,
        #                 "rate": 1,
        #             },
        #             "debug": False,
        #             "aggressiveness": 0,
        #             "max_episode_steps": 300,
        #         },
        #     },
        #     "000": {  # RL Agent uses too much GPU memory and crashes.
        #         "locator": "rl_agent:rl-agent-v0",
        #         "spec_train_params": {
        #             "goal_is_nearby_threshold": 40,
        #             "lane_end_threshold": 51,
        #             "lane_crash_distance_threshold": 6,
        #             "lane_crash_ttc_threshold": 2,
        #             "intersection_crash_distance_threshold": 6,
        #             "intersection_crash_ttc_threshold": 5,
        #             "target_speed": 15,
        #             "lane_change_speed": 10.0
        #         },
        #         "spec_eval_params": {
        #             "goal_is_nearby_threshold": 40,
        #             "lane_end_threshold": 51,
        #             "lane_crash_distance_threshold": 6,
        #             "lane_crash_ttc_threshold": 2,
        #             "intersection_crash_distance_threshold": 6,
        #             "intersection_crash_ttc_threshold": 5,
        #             "target_speed": 15,
        #             "lane_change_speed": 20.0
        #         },
        #     },
        #     "000": {
        #         "locator": "policies.non_interactive_agent:non-interactive-agent-v0",
        #         "spec_train_params": {
        #             "speed": 5,
        #             "target_lane_index": None,
        #         },
        #         "spec_eval_params": {
        #             "speed": 20,
        #             "target_lane_index": None,
        #         },
        #     },
        #     "000": {
        #         "locator": "policies.keep_lane_agent:keep-lane-agent-v0",
        #         "spec_train_params": {},
        #         "spec_eval_params": {},
        #     },
        # },
        # TODO: Get rid of max_episode_steps argument. Each agent can take their own
        #       max_episode_steps argument if they want.
        max_episode_steps=int(args.max_episode_steps),
        eval_info={
            "eval_rate": float(args.eval_rate),
            "eval_episodes": int(args.eval_episodes),
        },
        timestep_sec=float(args.timestep),
        headless=args.headless,
        seed=args.seed,
        log_dir=args.log_dir,
    )
