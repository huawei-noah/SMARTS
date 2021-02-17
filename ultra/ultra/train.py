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
import os, sys
import json
import os
from ultra.utils.ray import default_ray_kwargs

# Set environment to better support Ray
os.environ["MKL_NUM_THREADS"] = "1"
import time
import psutil, pickle, dill
import gym, ray, torch, argparse
from smarts.zoo.registry import make
from ultra.utils.episode import episodes
from ultra.evaluate import evaluation_check

num_gpus = 1 if torch.cuda.is_available() else 0


# @ray.remote(num_gpus=num_gpus / 2, max_calls=1)
@ray.remote(num_gpus=num_gpus / 2)
def train(
    scenario_info,
    num_episodes,
    policy_classes,
    eval_info,
    timestep_sec,
    headless,
    seed,
    log_dir,
):
    torch.set_num_threads(1)
    total_step = 0
    finished = False

    # E.g. From a ["ultra.baselines.dqn:dqn-v0", "ultra.baselines.ppo:ppo-v0"]
    # policy_classes list, transform it to an etag of "dqn-v0:ppo-v0".
    etag = ":".join(
        [policy_class.split(":")[-1] for policy_class in policy_classes]
    )

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
        agent_id: make(locator=policy_class)
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

    for episode in episodes(num_episodes, etag=etag, log_dir=log_dir):
        # Reset the environment and retrieve the initial observations.
        observations = env.reset()
        dones = {"__all__": False}
        infos = None
        episode.reset()
        experiment_dir = episode.experiment_dir

        # Save the entire spec (policy_params, reward_adapter, and observation_adapter).
        for agent_id, agent_spec in agent_specs.items():
            if not os.path.exists(f"{experiment_dir}/spec{agent_id}.pkl"):
                if not os.path.exists(experiment_dir):
                    os.makedirs(experiment_dir)
                with open(f"{experiment_dir}/spec{agent_id}.pkl", "wb") as spec_output:
                    dill.dump(agent_spec, spec_output, pickle.HIGHEST_PROTOCOL)

        while not dones["__all__"]:
            # Break if any of the agent's step counts is 1000000 or greater.
            if any([episode.get_itr(agent_id) >= 1000000 for agent_id in agents]):
                finished = True
                break

            # Perform the evaluation check for each agent.
            for agent_id, agent in agents.items():
                evaluation_check(
                    agent=agent,
                    agent_id=agent_id,
                    policy_class=agent_classes[agent_id],
                    episode=episode,
                    log_dir=log_dir,
                    **eval_info,
                    **env.info,
                )

            # Get and perform the available agents' actions.
            actions = {
                agent_id: agents[agent_id].act(observation, explore=True)
                for agent_id, observation in observations.items()
            }
            next_observations, rewards, dones, infos = env.step(actions)

            # Step and record the data of each available agent.
            for agent_id in observations.keys() & next_observations.keys():
                loss_output = agent.step(
                    state=observations[agent_id],
                    action=actions[agent_id],
                    reward=rewards[agent_id],
                    next_state=next_observations[agent_id],
                    done=dones[agent_id],
                )
                # episode.record_step(
                #     agent_id=agent_id,
                #     infos=infos,
                #     rewards=rewards,
                #     total_step=total_step,
                #     loss_output=loss_output,
                # )

            # Update variables for the next step.
            total_step += 1
            observations = next_observations

        # episode.record_episode()
        # episode.record_tensorboard(agent_id=AGENT_ID)

        if finished:
            break

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("intersection-single-agent")
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
        help="Policies available : [ppo, sac, ddpg, dqn, bdqn]",
        type=str,
        default="sac",
    )
    parser.add_argument(
        "--episodes", help="Number of training episodes", type=int, default=1000000
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
        "--seed", help="Environment seed", default=2, type=int,
    )
    parser.add_argument(
        "--log-dir", help="Log directory location", default="logs", type=str,
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
                    data["agents"][policy]["path"] + ":" + data["agents"][policy]["locator"]
                )
            else:
                raise ImportError("Invalid policy name. Please try again")

    ray.init()
    ray.wait(
        [
            train.remote(
                scenario_info=(args.task, args.level),
                num_episodes=int(args.episodes),
                eval_info={
                    "eval_rate": float(args.eval_rate),
                    "eval_episodes": int(args.eval_episodes),
                },
                timestep_sec=float(args.timestep),
                headless=args.headless,
                policy_classes=policy_classes,
                seed=args.seed,
                log_dir=args.log_dir,
            )
        ]
    )
