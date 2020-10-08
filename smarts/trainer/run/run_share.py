import argparse
from pathlib import Path
from ray import tune

from smarts.core.utils.file import copy_tree
from smarts.env.rllib_hiway_env import RLlibHiWayEnv
from smarts.trainer.zoo.tuned.tuned_space import rllib_trainable_agent
from smarts.trainer.zoo.tuned.callback import (
    on_episode_start,
    on_episode_step,
    on_episode_end,
)

RUN_NAME = Path(__file__).stem
EXPERIMENT_NAME_FMT = "{scenario}-{algorithm}-{n_agent}"


def parse_args():
    parser = argparse.ArgumentParser("share parameter learning")

    # env setting
    parser.add_argument("--scenario", type=str, default="loop", help="Scenario name")
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
    parser.add_argument("--num_workers", type=int, default=4, help="rllib num workers")
    parser.add_argument(
        "--horizon", type=int, default=1000, help="horizon for a episode"
    )
    parser.add_argument(
        "--resume", default=False, action="store_true", help="Resume training or not."
    )
    parser.add_argument(
        "--debug", default=False, action="store_true", help="Debug smarts or not"
    )
    # not work in multi trial
    parser.add_argument(
        "--restore",
        default=None,
        type=str,
        help="path to restore checkpoint, absolute dir",
    )
    parser.add_argument(
        "--log_dir",
        default="results",
        type=str,
        help="path to store checkpoint model, relative dir",
    )

    return parser.parse_args()


def main(args):
    # ====================================
    # init env config
    # ====================================
    rllib_agents = {
        f"AGENT-{i}": rllib_trainable_agent(continuous_action=args.algorithm in ["PPO"])
        for i in range(args.num_agents)
    }

    scenario_path = Path(args.scenario).absolute()
    env_config = {
        "seed": 42,
        "scenarios": [str(scenario_path)],
        "headless": args.headless,
        "agent_specs": {
            agent_id: rllib_agent["agent_spec"]
            for agent_id, rllib_agent in rllib_agents.items()
        },
    }

    # ====================================
    # init tune config
    # ====================================
    tune_config = {
        "env": RLlibHiWayEnv,
        "env_config": env_config,
        "multiagent": {
            "policies": {
                "default_policy": (
                    None,
                    rllib_agents["AGENT-0"]["observation_space"],
                    rllib_agents["AGENT-0"]["action_space"],
                    {},
                )
            },
            "policy_mapping_fn": lambda agent_id: "default_policy",
        },
        "callbacks": {
            "on_episode_start": on_episode_start,
            "on_episode_step": on_episode_step,
            "on_episode_end": on_episode_end,
        },
        "lr": 1e-4,
        "log_level": "WARN",
        "num_workers": args.num_workers,
        "horizon": args.horizon,
        "train_batch_size": 10240 * 3,
    }

    if args.algorithm == "PPO":
        tune_config.update(
            {
                "lambda": 0.95,
                "clip_param": 0.2,
                "num_sgd_iter": 10,
                "sgd_minibatch_size": 1024,
            }
        )
    elif args.algorithm in ["A2C", "A3C"]:
        tune_config.update({"lambda": 0.95})

    # ====================================
    # init log and checkpoint dir_info
    # ====================================
    experiment_name = EXPERIMENT_NAME_FMT.format(
        scenario=args.scenario, algorithm=args.algorithm, n_agent=args.num_agents,
    )

    log_dir = Path(args.log_dir).expanduser().absolute() / RUN_NAME
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpointing at {log_dir}")

    if args.restore:
        restore_path = Path(args.restore).expanduser()
        print(f"Loading model from {restore_path}")
    else:
        restore_path = None

    # run experiments
    analysis = tune.run(
        args.algorithm,
        name=experiment_name,
        stop={"time_total_s": 24 * 60 * 60},
        checkpoint_freq=10,
        checkpoint_at_end=True,
        local_dir=str(log_dir),
        resume=args.resume,
        restore=restore_path,
        max_failures=1000,
        export_formats=["model", "checkpoint"],
        config=tune_config,
    )

    print(analysis.dataframe().head())

    max_reward_logdir = analysis.get_best_logdir("episode_reward_max")
    model_path = Path(max_reward_logdir) / "model"
    dest_model_path = log_dir / "model"

    copy_tree(model_path, dest_model_path, overwrite=True)
    print(f"Wrote model to: {dest_model_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
