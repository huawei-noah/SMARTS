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
import json

# Set environment to better support Ray
os.environ["MKL_NUM_THREADS"] = "1"
import glob
import yaml
import time
import numpy as np
import gym, ray, torch, argparse
import psutil
from pydoc import locate
from ultra.utils.episode import LogInfo, episodes
from ultra.utils.ray import default_ray_kwargs
from smarts.zoo.registry import make

num_gpus = 1 if torch.cuda.is_available() else 0


def evaluation_check(
    agent,
    episode,
    agent_id,
    policy_class,
    eval_rate,
    eval_episodes,
    scenario_info,
    timestep_sec,
    headless,
):
    agent_itr = episode.get_itr(agent_id)

    if (agent_itr + 1) % eval_rate == 0 and episode.last_eval_iteration != agent_itr:
        checkpoint_dir = episode.checkpoint_dir(agent_itr)
        agent.save(checkpoint_dir)
        episode.eval_mode()
        episode.info[episode.active_tag] = ray.get(
            [
                evaluate.remote(
                    experiment_dir=episode.experiment_dir,
                    agent_id="AGENT_008",
                    policy_class=policy_class,
                    seed=episode.eval_count,
                    itr_count=agent_itr,
                    checkpoint_dir=checkpoint_dir,
                    scenario_info=scenario_info,
                    num_episodes=eval_episodes,
                    headless=headless,
                    timestep_sec=timestep_sec,
                )
            ]
        )[0]
        episode.eval_count += 1
        episode.last_eval_iteration = agent_itr
        episode.record_tensorboard(agent_id=agent_id)
        episode.train_mode()


# Number of GPUs should be splited between remote functions.
@ray.remote(num_gpus=num_gpus / 2)
def evaluate(
    experiment_dir,
    seed,
    agent_id,
    policy_class,
    itr_count,
    checkpoint_dir,
    scenario_info,
    num_episodes,
    headless,
    timestep_sec,
):

    torch.set_num_threads(1)
    spec = make(
        locator=policy_class,
        checkpoint_dir=checkpoint_dir,
        experiment_dir=experiment_dir,
    )

    env = gym.make(
        "ultra.env:ultra-v0",
        agent_specs={agent_id: spec},
        scenario_info=scenario_info,
        headless=headless,
        timestep_sec=timestep_sec,
        seed=seed,
        eval_mode=True,
    )

    agent = spec.build_agent()
    summary_log = LogInfo()
    logs = []

    for episode in episodes(num_episodes):
        observations = env.reset()
        state = observations[agent_id]
        dones, infos = {"__all__": False}, None

        episode.reset(mode="Evaluation")
        while not dones["__all__"]:
            action = agent.act(state, explore=False)
            observations, rewards, dones, infos = env.step({agent_id: action})

            next_state = observations[agent_id]

            state = next_state

            episode.record_step(agent_id=agent_id, infos=infos, rewards=rewards)

        episode.record_episode()
        logs.append(episode.info[episode.active_tag].data)

        for key, value in episode.info[episode.active_tag].data.items():
            if not isinstance(value, (list, tuple, np.ndarray)):
                summary_log.data[key] += value

    for key, val in summary_log.data.items():
        if not isinstance(val, (list, tuple, np.ndarray)):
            summary_log.data[key] /= num_episodes

    env.close()

    return summary_log


if __name__ == "__main__":
    parser = argparse.ArgumentParser("intersection-single-agent-evaluation")
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
    parser.add_argument("--models", default="models/", help="Directory to saved models")
    parser.add_argument(
        "--episodes", help="Number of training episodes", type=int, default=200
    )
    parser.add_argument(
        "--timestep", help="Environment timestep (sec)", type=float, default=0.1
    )
    parser.add_argument(
        "--headless", help="Run without envision", type=bool, default=False
    )
    parser.add_argument(
        "--experiment-dir",
        help="Path to spec file that includes adapters and policy parameters",
        type=str,
    )
    args = parser.parse_args()

    # --------------------------------------------------------
    if not os.path.exists(args.models):
        raise "Models not Found"

    sorted_models = sorted(
        glob.glob(f"{args.models}/*"), key=lambda x: int(x.split("/")[-1])
    )

    with open("ultra/agent_pool.json", "r") as f:
        data = json.load(f)
        if args.policy in data["agents"].keys():
            policy_path = data["agents"][args.policy]["path"]
            policy_locator = data["agents"][args.policy]["locator"]
        else:
            raise ImportError("Invalid policy name. Please try again")

    # Required string for smarts' class registry
    policy_class = str(policy_path) + ":" + str(policy_locator)
    num_cpus = max(
        1, psutil.cpu_count(logical=False) - 1
    )  # remove `logical=False` to use all cpus
    ray_kwargs = default_ray_kwargs(num_cpus=num_cpus, num_gpus=num_gpus)
    ray.init(**ray_kwargs)
    try:
        agent_id = "AGENT_008"
        for episode in episodes(len(sorted_models), etag=args.policy):
            model = sorted_models[episode.index]
            print("model: ", model)
            episode_count = model.split("/")[-1]
            episode.eval_mode()
            episode.info[episode.active_tag] = ray.get(
                [
                    evaluate.remote(
                        experiment_dir=args.experiment_dir,
                        agent_id=agent_id,
                        policy_class=policy_class,
                        seed=episode.eval_count,
                        itr_count=0,
                        checkpoint_dir=model,
                        scenario_info=(args.task, args.level),
                        num_episodes=int(args.episodes),
                        timestep_sec=float(args.timestep),
                        headless=args.headless,
                    )
                ]
            )[0]
            episode.record_tensorboard(agent_id=agent_id)
            episode.eval_count += 1
    finally:
        time.sleep(1)
        ray.shutdown()
