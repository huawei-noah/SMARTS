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
import os
import pickle

# Set environment to better support Ray
os.environ["MKL_NUM_THREADS"] = "1"
import argparse
import glob
import time
from pydoc import locate

import gym
import numpy as np
import ray
import torch

from smarts.zoo.registry import make
from ultra.utils.episode import LogInfo, episodes
from ultra.utils.ray import default_ray_kwargs

num_gpus = 1 if torch.cuda.is_available() else 0


def evaluation_check(
    agents,
    episode,
    agent_ids,
    policy_classes,
    eval_rate,
    eval_episodes,
    max_episode_steps,
    scenario_info,
    timestep_sec,
    headless,
    log_dir,
):
    # Evaluate agents that have reached the eval_rate.
    agent_ids_to_evaluate = [
        agent_id
        for agent_id in agent_ids
        if (episode.get_itr(agent_id) + 1) % eval_rate == 0
        and episode.last_eval_iterations[agent_id] != episode.get_itr(agent_id)
    ]

    # Skip evaluation if there are no agents needing an evaluation.
    if len(agent_ids_to_evaluate) < 1:
        return

    episode.eval_mode()
    evaluation_data = {}

    for agent_id in agent_ids_to_evaluate:
        # Get the checkpoint directory for the current agent and save its model.
        checkpoint_directory = episode.checkpoint_dir(
            agent_id, episode.get_itr(agent_id)
        )
        agents[agent_id].save(checkpoint_directory)

        # Perform the evaluation on this agent and save the data.
        evaluation_data.update(
            ray.get(
                [
                    evaluate.remote(
                        seed=episode.eval_count,
                        experiment_dir=episode.experiment_dir,
                        agent_ids=[agent_id],
                        policy_classes={agent_id: policy_classes[agent_id]},
                        checkpoint_dirs={agent_id: checkpoint_directory},
                        scenario_info=scenario_info,
                        num_episodes=eval_episodes,
                        max_episode_steps=max_episode_steps,
                        headless=headless,
                        timestep_sec=timestep_sec,
                        log_dir=log_dir,
                    )
                ]
            )[0]
        )
        episode.eval_count += 1
        episode.last_eval_iterations[agent_id] = episode.get_itr(agent_id)

    # Put the evaluation data for all agents into the episode and record the TensorBoard.
    episode.info[episode.active_tag] = evaluation_data
    episode.record_tensorboard()
    episode.train_mode()


# Number of GPUs should be split between remote functions.
@ray.remote(num_gpus=num_gpus / 2)
def evaluate(
    experiment_dir,
    seed,
    agent_ids,
    policy_classes,
    checkpoint_dirs,
    scenario_info,
    num_episodes,
    max_episode_steps,
    headless,
    timestep_sec,
    log_dir,
):
    torch.set_num_threads(1)

    # Create the agent specifications matched with their associated ID.
    agent_specs = {
        agent_id: make(
            locator=policy_classes[agent_id],
            checkpoint_dir=checkpoint_dirs[agent_id],
            experiment_dir=experiment_dir,
            max_episode_steps=max_episode_steps,
            agent_id=agent_id,
        )
        for agent_id in agent_ids
    }

    # Create the environment with the specified agents.
    env = gym.make(
        "ultra.env:ultra-v0",
        agent_specs=agent_specs,
        scenario_info=scenario_info,
        headless=headless,
        timestep_sec=timestep_sec,
        seed=seed,
        eval_mode=True,
    )

    # Build each agent from its specification.
    agents = {
        agent_id: agent_spec.build_agent()
        for agent_id, agent_spec in agent_specs.items()
    }

    # A dictionary to hold the evaluation data for each agent.
    summary_log = {agent_id: LogInfo() for agent_id in agent_ids}

    # Define an 'etag' for this experiment's data directory based off policy_classes.
    # E.g. From a ["ultra.baselines.dqn:dqn-v0", "ultra.baselines.ppo:ppo-v0"]
    # policy_classes list, transform it to an etag of "dqn-v0:ppo-v0".
    etag = ":".join([policy_class.split(":")[-1] for policy_class in policy_classes])

    for episode in episodes(num_episodes, etag=etag, log_dir=log_dir):
        # Reset the environment and retrieve the initial observations.
        observations = env.reset()
        dones = {"__all__": False}
        infos = None
        episode.reset(mode="Evaluation")

        while not dones["__all__"]:
            # Get and perform the available agents' actions.
            actions = {
                agent_id: agents[agent_id].act(observation, explore=False)
                for agent_id, observation in observations.items()
            }
            observations, rewards, dones, infos = env.step(actions)

            # Record the data from this episode.
            episode.record_step(
                agent_ids_to_record=infos.keys(), infos=infos, rewards=rewards
            )

        episode.record_episode()

        for agent_id, agent_data in episode.info[episode.active_tag].items():
            for key, value in agent_data.data.items():
                if not isinstance(value, (list, tuple, np.ndarray)):
                    summary_log[agent_id].data[key] += value

    # Normalize by the number of evaluation episodes.
    for agent_id, agent_data in summary_log.items():
        for key, value in agent_data.data.items():
            if not isinstance(value, (list, tuple, np.ndarray)):
                summary_log[agent_id].data[key] /= num_episodes

    env.close()

    return summary_log


if __name__ == "__main__":
    parser = argparse.ArgumentParser("intersection-evaluation")
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
        "--models",
        nargs="+",
        help="The models in the models/ directory of the experiment to evaluate",
    )
    parser.add_argument(
        "--episodes", help="Number of training episodes", type=int, default=200
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
        "--experiment-dir",
        help="Path to the experiment directory",
        type=str,
    )
    parser.add_argument(
        "--log-dir",
        help="Log directory location",
        default="logs",
        type=str,
    )
    args = parser.parse_args()

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    if not all([os.path.exists(model_path) for model_path in args.models]):
        raise "At least one path to a model is invalid"

    if not all([os.listdir(model_path) for model_path in args.models]):
        raise "There are no models to evaluate in at least one model path"

    agent_ids_from_models = [
        os.path.basename(os.path.normpath(model_path)) for model_path in args.models
    ]

    # Load relevant agent metadata.
    with open(
        os.path.join(args.experiment_dir, "agent_metadata.pkl"), "rb"
    ) as metadata_file:
        agent_metadata = pickle.load(metadata_file)

    # Extract the agent IDs and policy classes from the metadata and given models.
    agent_ids = [
        agent_id
        for agent_id in agent_metadata["agent_ids"]
        if agent_id in agent_ids_from_models
    ]
    policy_classes = {
        agent_id: agent_metadata["agent_classes"][agent_id] for agent_id in agent_ids
    }

    # From a base model directory such as logs/<experiment_name>/models/*, assign each agent ID with its
    # checkpoint directories in sorted order based on the checkpoint iteration. The agent IDs are
    # obtained from the direct child folders of the model directory given. As an example result:
    # {
    #     '000': ['logs/<experiment_name>/models/000/1042', 'logs/<experiment_name>/models/000/2062'],
    #     '001': ['logs/<experiment_name>/models/001/999', 'logs/<experiment_name>/models/001/1999'],
    #     '003': ['logs/<experiment_name>/models/003/1009', 'logs/<experiment_name>/models/003/2120'],
    #     '002': ['logs/<experiment_name>/models/002/1053', 'logs/<experiment_name>/models/002/2041'],
    # }
    agent_checkpoint_directories = {
        agent_id: sorted(
            glob.glob(os.path.join(args.experiment_dir, "models", agent_id, "*")),
            key=lambda x: int(x.split("/")[-1]),
        )
        for agent_id in agent_ids
    }

    # Assert each agent ID has the same number of checkpoints saved.
    directories_iterator = iter(agent_checkpoint_directories.values())
    number_of_checkpoints = len(next(directories_iterator))
    assert all(
        len(checkpoint_directory) == number_of_checkpoints
        for checkpoint_directory in directories_iterator
    ), "Not all agents have the same number of checkpoints saved"

    # Define an 'etag' for this experiment's data directory based off policy_classes.
    # E.g. From a {"000": "ultra.baselines.dqn:dqn-v0", "001": "ultra.baselines.ppo:ppo-v0"]
    # policy_classes dict, transform it to an etag of "dqn-v0:ppo-v0-evaluation".
    etag = (
        ":".join([policy_classes[agent_id].split(":")[-1] for agent_id in agent_ids])
        + "-evaluation"
    )

    ray.init()
    try:
        for episode in episodes(
            number_of_checkpoints,
            etag=etag,
            log_dir=args.log_dir,
        ):
            # Obtain a checkpoint directory for each agent.
            current_checkpoint_directories = {
                agent_id: agent_directories[episode.index]
                for agent_id, agent_directories in agent_checkpoint_directories.items()
            }
            episode.eval_mode()
            episode.info[episode.active_tag] = ray.get(
                [
                    evaluate.remote(
                        experiment_dir=args.experiment_dir,
                        agent_ids=agent_ids,
                        policy_classes=policy_classes,
                        seed=episode.eval_count,
                        checkpoint_dirs=current_checkpoint_directories,
                        scenario_info=(args.task, args.level),
                        num_episodes=int(args.episodes),
                        max_episode_steps=int(args.max_episode_steps),
                        timestep_sec=float(args.timestep),
                        headless=args.headless,
                        log_dir=args.log_dir,
                    )
                ]
            )[0]
            episode.record_tensorboard()
            episode.eval_count += 1
    finally:
        time.sleep(1)
        ray.shutdown()
