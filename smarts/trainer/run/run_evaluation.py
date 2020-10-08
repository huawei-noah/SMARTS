import argparse
import time

from pathlib import Path

import ray
from ray.rllib.env.env_context import EnvContext

from smarts.env.rllib_hiway_env import RLlibHiWayEnv
from smarts.trainer.zoo.tuned.tuned_space import agent_spec_from_model


def parse_args():
    parser = argparse.ArgumentParser("run evaluation for rendering")
    # env setting
    parser.add_argument("--scenario", required=True, type=str, help="Path to scenario")
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
        "--load_path", "-p", required=True, type=str, help="path to restore model",
    )
    parser.add_argument(
        "--policy_name",
        "-n",
        default="default_policy",
        type=str,
        help="policy name training used",
    )
    args = parser.parse_args()
    return args


def main(args):
    # Since we are only evaluating a single model, use the same AgentSpec for all agents
    common_agent_spec = agent_spec_from_model(
        model_path=Path(args.load_path).expanduser().resolve().absolute(),
        algorithm=args.algorithm,
        continuous=args.algorithm in ["PPO"],
        model_type=args.load_type,
    )
    agent_specs = {f"AGENT-{i}": common_agent_spec for i in range(args.num_agents)}
    scenario_path = Path(args.scenario).expanduser().resolve().absolute()

    ray.init()

    env = RLlibHiWayEnv(
        config=EnvContext(
            {
                "seed": 42,
                "scenarios": [str(scenario_path)],
                "headless": args.headless,
                "agent_specs": agent_specs,
            },
            worker_index=0,
        )
    )

    total_steps = 0
    total_infer_time = 0
    total_step_time = 0
    total_time = 0

    for i in range(10):
        env_obs = env.reset()
        agents = {
            agent_id: spec.build_agent() for agent_id, spec in agent_specs.items()
        }

        for _ in range(1000):
            t0 = time.time()
            t1 = time.time()
            actions = {
                agent_id: agents[agent_id].act(obs) for agent_id, obs in env_obs.items()
            }
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


if __name__ == "__main__":
    args = parse_args()
    main(args)
