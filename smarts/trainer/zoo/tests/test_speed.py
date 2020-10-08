import argparse
import os
import time

import ray

# TODO: fix this test

from smarts.env.rllib_hiway_env import RLlibHiWayEnv
from smarts.trainer.zoo.tuned.tuned_space import rllib_trainable_agent
from smarts.trainer.zoo.tuned.default_model import (
    BatchRLLibTFSavedModelPolicy,
    BatchRLLibTFCheckpointPolicy,
)

TRAINER_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
BASE_DIR = os.path.dirname(os.path.dirname(TRAINER_DIR))
EXPERIMENT_NAME = "{scenario}-{algorithm}-{n_agent}"


def parse_args():
    parser = argparse.ArgumentParser("run evaluation for rendering")
    # env setting
    parser.add_argument("--scenario", type=str, default="loop", help="scenario name")
    parser.add_argument("--num_agents", type=int, default=1, help="Agent number")
    parser.add_argument(
        "--headless", default=False, action="store_true", help="Turn on headless mode"
    )

    # training setting
    parser.add_argument(
        "--algorithm",
        type=str,
        default="PPO",
        help="training algorithms",
        choices=["PPO", "A2C", "A3C", "PG", "DQN"],
    )
    parser.add_argument(
        "--debug", default=False, action="store_true", help="Turn on headless mode"
    )
    parser.add_argument(
        "--load_type",
        "-t",
        default="model",
        type=str,
        help="restore type",
        choices=["checkpoint", "model"],
    )
    parser.add_argument(
        "--load_path", "-p", default=None, type=str, help="path to restore model",
    )
    parser.add_argument(
        "--policy_name",
        "-n",
        default="default_policy",
        type=str,
        help="policy name training used",
    )
    args = parser.parse_args()
    assert args.load_path is not None
    return args


def main(args):
    # here only use winner observation and reward.
    if args.algorithm == "PPO":
        agents = {
            f"AGENT-{i}": rllib_trainable_agent(continuous_action=True)
            for i in range(args.num_agents)
        }
    else:
        agents = {
            f"AGENT-{i}": rllib_trainable_agent(continuous_action=False)
            for i in range(args.num_agents)
        }

    scenario_root = os.path.join(BASE_DIR, "scenarios", args.scenario)

    ray.init()

    env = RLlibHiWayEnv(
        config={
            "seed": 42,
            "scenarios": [scenario_root],
            "headless": args.headless,
            "agents": agents,
        }
    )

    if args.load_type == "model":
        agent_policy = BatchRLLibTFSavedModelPolicy(
            args.load_path,
            args.algorithm,
            args.policy_name,
            agents["AGENT-0"].observation_space,
        )
    elif args.load_type == "checkpoint":
        agent_policy = BatchRLLibTFCheckpointPolicy(
            args.load_path,
            args.algorithm,
            args.policy_name,
            agents["AGENT-0"].observation_space,
            agents["AGENT-0"].action_space,
        )
    else:
        raise TypeError("Unsupport load type")

    agent_policy.setup()
    total_steps = 0
    total_infer_time = 0
    total_step_time = 0
    total_time = 0

    for i in range(10):
        env_obs = env.reset()

        for _ in range(10):
            t0 = time.time()
            t1 = time.time()
            actions = agent_policy.act(env_obs)
            total_infer_time += time.time() - t1

            t2 = time.time()
            env_obs, env_reward, done, _ = env.step(actions)
            total_step_time += time.time() - t2

            total_steps += 1

            if done["__all__"]:
                print("simulation ended")
                break
            total_time += time.time() - t0

        print(
            f"total_steps:{total_steps}, total_infer_time: {total_infer_time}s, total_step_time: {total_step_time}s,"
            f"total_time:{total_time}s, avg_infer_time:{total_infer_time/total_steps}s, "
            f"avg_step_time:{total_step_time/total_steps}s, avg_total_time:{total_time/total_steps}s"
        )

    env.close()
    agent_policy.teardown()


if __name__ == "__main__":
    args = parse_args()
    main(args)
