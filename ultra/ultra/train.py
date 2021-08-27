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
import glob
from ultra.utils.ray import default_ray_kwargs

# Set environment to better support Ray
os.environ["MKL_NUM_THREADS"] = "1"
import argparse
import dill
import gym
import pickle
import ray
import time
import torch

from smarts.zoo.registry import make
from ultra.evaluate import evaluation_check, collect_evaluations
from ultra.utils.common import agent_pool_value
from ultra.utils.episode import episodes

num_gpus = 1 if torch.cuda.is_available() else 0


def create_argument_parser():
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
        default=1200,
    )
    parser.add_argument(
        "--max-steps",
        help="Maximum total number of training steps",
        type=int,
        default=1000000,
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
    parser.add_argument(
        "--experiment-dir",
        help="Path to the base dir of trained model",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--policy-ids",
        help="Name of each specified policy",
        default=None,
        type=str,
    )
    return parser


def load_agents(experiment_dir):
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

    latest_checkpoint_directories = {}
    for agent_id in agent_ids:
        checkpoint_directories = sorted(
            glob.glob(os.path.join(experiment_dir, "models", agent_id, "*")),
            key=lambda x: int(x.split("/")[-1]),
        )
        latest_checkpoint_directory = (
            checkpoint_directories[-1] if len(checkpoint_directories) > 0 else None
        )
        latest_checkpoint_directories[agent_id] = latest_checkpoint_directory

    # Create the agent specifications matched with their associated ID and corresponding
    # checkpoint directory
    agent_specs = {
        agent_id: make(
            locator=agent_classes[agent_id],
            checkpoint_dir=latest_checkpoint_directories[agent_id],
            experiment_dir=experiment_dir,
            agent_id=agent_id,
        )
        for agent_id in agent_ids
    }

    # Create the agents matched with their associated ID.
    agents = {
        agent_id: agent_spec.build_agent()
        for agent_id, agent_spec in agent_specs.items()
    }

    # Load any extra data that the agent needs from its last training run in order to
    # resume training (e.g. its previous replay buffer experience).
    for agent_id, agent in agents.items():
        extras_directory = os.path.join(experiment_dir, "extras", agent_id)
        agent.load_extras(extras_directory)

    return agent_ids, agent_classes, agent_specs, agents


def build_agents(policy_classes, policy_ids, max_episode_steps):
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

    return agent_ids, agent_classes, agent_specs, agents


def _save_agent_metadata(
    experiment_dir, filename, agent_ids, agent_classes, agent_specs
):
    # Save relevant agent metadata.
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)
    with open(os.path.join(experiment_dir, filename), "wb") as metadata_file:
        dill.dump(
            {
                "agent_ids": agent_ids,
                "agent_classes": agent_classes,
                "agent_specs": agent_specs,
            },
            metadata_file,
            pickle.HIGHEST_PROTOCOL,
        )


def train(
    scenario_info,
    num_episodes,
    policy_classes,
    max_episode_steps,
    max_steps,
    eval_info,
    timestep_sec,
    headless,
    seed,
    log_dir,
    experiment_dir=None,
    policy_ids=None,
):
    torch.set_num_threads(1)
    total_step = 0
    finished = False
    evaluation_task_ids = dict()

    if experiment_dir:
        agent_ids, agent_classes, agent_specs, agents = load_agents(experiment_dir)
    else:
        agent_ids, agent_classes, agent_specs, agents = build_agents(
            policy_classes, policy_ids, max_episode_steps
        )

    # Define an 'etag' for this experiment's data directory based off policy_classes.
    # E.g. From a ["ultra.baselines.dqn:dqn-v0", "ultra.baselines.ppo:ppo-v0"]
    # policy_classes list, transform it to an etag of "dqn-v0:ppo-v0".
    etag = ":".join(
        [agent_class.split(":")[-1] for agent_class in agent_classes.values()]
    )

    # Create the environment.
    env = gym.make(
        "ultra.env:ultra-v0",
        agent_specs=agent_specs,
        scenario_info=scenario_info,
        headless=headless,
        timestep_sec=timestep_sec,
        seed=seed,
    )

    for episode in episodes(num_episodes, etag=etag, log_dir=log_dir):

        # Reset the environment and retrieve the initial observations.
        observations = env.reset()
        scenario_name = env.scenario_log["scenario_map"]
        dones = {"__all__": False}
        infos = None
        episode.reset()

        experiment_dir = episode.experiment_dir
        # Name of agent metadata pickle file
        filename = "agent_metadata.pkl"
        if not os.path.exists(os.path.join(experiment_dir, filename)):
            _save_agent_metadata(
                experiment_dir,
                filename,
                agent_ids,
                agent_classes,
                agent_specs,
            )

        evaluation_check(
            agents=agents,
            agent_ids=agent_ids,
            policy_classes=agent_classes,
            episode=episode,
            log_dir=log_dir,
            max_episode_steps=max_episode_steps,
            evaluation_task_ids=evaluation_task_ids,
            **eval_info,
            **env.info,
        )

        collect_evaluations(evaluation_task_ids=evaluation_task_ids)

        while not dones["__all__"]:
            # Break if any of the agent's step counts is max_steps (default is 1000000) or greater.
            if any([episode.get_itr(agent_id) >= max_steps for agent_id in agents]):
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

        episode.record_episode(scenario_name)
        episode.record_tensorboard(recording_step=episode.index)

        if finished:
            break

    # Save any extra data that the agent may need to resume training in a future
    # training session (e.g. its replay buffer experience).
    for agent_id, agent in agents.items():
        extras_directory = episode.extras_dir(agent_id)
        agent.save_extras(extras_directory)

    # Wait on the remaining evaluations to finish.
    while collect_evaluations(evaluation_task_ids):
        time.sleep(0.1)

    env.close()


if __name__ == "__main__":
    parser = create_argument_parser()
    args = parser.parse_args()

    # Obtain the policy class strings for each specified policy.
    policy_classes = [
        agent_pool_value(agent_name, "policy_class")
        for agent_name in args.policy.split(",")
    ]

    # Obtain the policy class IDs from the arguments.
    policy_ids = args.policy_ids.split(",") if args.policy_ids else None

    ray.init()
    try:
        train(
            scenario_info=(args.task, args.level),
            num_episodes=int(args.episodes),
            max_episode_steps=int(args.max_episode_steps),
            max_steps=int(args.max_steps),
            eval_info={
                "eval_rate": float(args.eval_rate),
                "eval_episodes": int(args.eval_episodes),
            },
            timestep_sec=float(args.timestep),
            headless=args.headless,
            policy_classes=policy_classes,
            seed=args.seed,
            log_dir=args.log_dir,
            experiment_dir=args.experiment_dir,
            policy_ids=policy_ids,
        )
    finally:
        ray.shutdown()
