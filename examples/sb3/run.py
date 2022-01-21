import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Silence the TF logs

import argparse
import pathlib
from datetime import datetime
from shutil import copyfile

from ruamel.yaml import YAML
from sb3.env.make_env import make_env
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# import warnings
# warnings.simplefilter("ignore", category=DeprecationWarning)
# warnings.simplefilter("ignore", category=PendingDeprecationWarning)
# warnings.filterwarnings("ignore",  ".*Vehicle.*")

yaml = YAML(typ="safe")


def main(args):
    # Load config file.
    config_file = yaml.load(
        (pathlib.Path(__file__).absolute().parent / "config.yaml").read_text()
    )

    # Load SMARTS env config.
    config_env = config_file["smarts"]
    config_env["mode"] = args.mode
    config_env["headless"] = not args.head
    config_env["scenarios_dir"] = (
        pathlib.Path(__file__).absolute().parents[0] / "scenarios"
    )
    _build_scenario()

    # Train or evaluate.
    if config_env["mode"] == "train" and not args.logdir:
        # Train from scratch.
        time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        logdir = pathlib.Path(__file__).absolute().parents[0] / "logs" / time
    elif config_env["mode"] == "train" and args.logdir:
        # Train from a pretrained model.
        logdir = args.logdir
    elif config_env["mode"] == "evaluate":
        logdir = args.logdir
    else:
        raise KeyError(
            f'Expected \'train\' or \'evaluate\', but got {config_env["mode"]}.'
        )
    config_env["logdir"] = args.logdir
    logdir = pathlib.Path(logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)
    print("Logdir:", logdir)

    # Run training or evaluation.
    run(config_env, logdir)


def _build_scenario():
    scenario = str(pathlib.Path(__file__).absolute().parent / "scenarios")
    build_scenario = f"scl scenario build-all --clean {scenario}"
    os.system(build_scenario)


def run(config, logdir):
    if config["mode"] == "evaluate":
        print("Start evaluation.")
        model = PPO.load(logdir / "model.zip")
        env = make_env(config)
        mean_reward, std_reward = evaluate_policy(
            model, env, n_eval_episodes=10, deterministic=True
        )
        print(f"Mean reward:{mean_reward:.2f} +/- {std_reward:.2f}")

    elif config["mode"] == "train" and args.logdir:
        print("Start training from existing model.")
        model = PPO.load(logdir / "model.zip")
        env = make_env(config)

        initial_mean_reward, initial_std_reward = evaluate_policy(
            model, env, n_eval_episodes=10, deterministic=True
        )
        model.set_env(env)
        model.learn(total_timesteps=args.train_steps)
        mean_reward, std_reward = evaluate_policy(
            model, env, n_eval_episodes=10, deterministic=True
        )
        print(
            f"Initial mean reward: {initial_mean_reward:.2f} +/- {initial_std_reward:.2f}"
        )
        print(f"Final mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

        # Save trained model.
        # copyfile(config_path, logdir / "config.yaml")
        model.save(logdir / "model")
        env.close()

    elif config["mode"] == "train" and not args.logdir:
        print("Start training.")
        env = make_env(config)
        model = PPO(
            "CnnPolicy",
            env,
            verbose=1,
            tensorboard_log=logdir / "tensorboard_log",
            use_sde=True,
        )

        initial_mean_reward, initial_std_reward = evaluate_policy(
            model, env, n_eval_episodes=10, deterministic=True
        )
        model.learn(total_timesteps=args.train_steps)
        mean_reward, std_reward = evaluate_policy(
            model, env, n_eval_episodes=10, deterministic=True
        )
        print(
            f"Initial mean reward: {initial_mean_reward:.2f} +/- {initial_std_reward:.2f}"
        )
        print(f"Final mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

        # Save trained model.
        # copyfile(config_path, str(save_path) + "/config.yaml")
        model.save(logdir / "model")
        env.close()

    else:
        raise KeyError(f'Expected \'train\' or \'evaluate\', but got {config["mode"]}.')


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
    # parser.add_argument("--train-steps", type=int, default=1e6)
    parser.add_argument(
        "--train-steps", help="Number of training steps.", type=int, default=1000
    )

    args = parser.parse_args()

    if args.mode == "evaluate" and args.logdir is None:
        raise Exception("When --mode=evaluate, --logdir option must be specified.")

    main(args)
