import argparse
import os
import pathlib
from datetime import datetime
from shutil import copyfile

from ruamel.yaml import YAML
from sb3.env.make_env import make_env
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

yaml = YAML(typ="safe")


def _build_scenario():
    scenario = str(pathlib.Path(__file__).absolute().parent / "scenarios")
    build_scenario = f"scl scenario build-all --clean {scenario}"
    os.system(build_scenario)


def main(args):
    # Load config file.
    config_file = yaml.load(
        (pathlib.Path(__file__).absolute().parent / "config.yaml").read_text()
    )
    _build_scenario()

    if args.mode != "colab":
        # Save trained model.
        time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        save_path = pathlib.Path(__file__).absolute().parent / "logs" / time
        pathlib.Path(str(save_path)).mkdir(parents=True, exist_ok=True)

    if args.mode == "evaluate":

        config_path = pathlib.Path(args.logdir) / "config.yaml"
        config_env = yaml.load((config_path).read_text())
        config_env = config_env["smarts"]
        config_env["headless"] = not args.head
        config_env["scenarios_dir"] = (
            pathlib.Path(__file__).absolute().parents[0] / "scenarios"
        )

        model = PPO.load(args.logdir + "/model.zip")
        env = make_env(config_env)
        mean_reward, std_reward = evaluate_policy(
            model, env, n_eval_episodes=10, deterministic=True
        )
        print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

    elif args.mode == "retrain":

        config_path = pathlib.Path(args.logdir) / "config.yaml"
        config_env = yaml.load((config_path).read_text())
        config_env = config_env["smarts"]
        config_env["headless"] = not args.head
        config_env["scenarios_dir"] = (
            pathlib.Path(__file__).absolute().parents[0] / "scenarios"
        )

        model = PPO.load(args.logdir + "/model.zip")
        env = make_env(config_env)

        before_mean_reward, before_std_reward = evaluate_policy(
            model, env, n_eval_episodes=10, deterministic=True
        )
        model.set_env(env)
        model.learn(total_timesteps=args.num_steps)
        mean_reward, std_reward = evaluate_policy(
            model, env, n_eval_episodes=10, deterministic=True
        )
        print(
            f"before_mean_reward:{before_mean_reward:.2f} +/- {before_std_reward:.2f}"
        )
        print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

        # Save trained model.
        copyfile(config_path, str(save_path) + "/config.yaml")
        model.save(str(save_path) + "/model")

    else:
        config_path = pathlib.Path(__file__).absolute().parent / "config.yaml"
        config_env = yaml.load((config_path).read_text())
        config_env = config_env["smarts"]
        config_env["headless"] = not args.head
        config_env["scenarios_dir"] = (
            pathlib.Path(__file__).absolute().parents[0] / "scenarios"
        )

        env = make_env(config_env)
        model = PPO(
            "CnnPolicy",
            env,
            verbose=1,
            tensorboard_log=str(save_path) + "/tensorboard_log",
            use_sde=True,
        )

        before_mean_reward, before_std_reward = evaluate_policy(
            model, env, n_eval_episodes=10, deterministic=True
        )
        model.learn(total_timesteps=args.num_steps)
        mean_reward, std_reward = evaluate_policy(
            model, env, n_eval_episodes=10, deterministic=True
        )
        print(
            f"before_mean_reward:{before_mean_reward:.2f} +/- {before_std_reward:.2f}"
        )
        print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

        # Save trained model.
        copyfile(config_path, str(save_path) + "/config.yaml")
        model.save(str(save_path) + "/model")


if __name__ == "__main__":
    program = pathlib.Path(__file__).stem
    parser = argparse.ArgumentParser(program)

    parser.add_argument(
        "--mode",
        help="`train` or `evaluate`. Default is `train`.",
        type=str,
        default="train",
    )
    parser.add_argument(
        "--logdir",
        help="Directory path to saved RL model. Required if `--mode=evaluate`, else optional.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--head", help="Run the simulation with display.", action="store_true"
    )
    parser.add_argument("--num-steps", type=int, default=1000000)

    args = parser.parse_args()

    if args.mode == "evaluate" and args.logdir is None:
        raise Exception("When --mode=evaluate, --logdir option must be specified.")

    main(args)
