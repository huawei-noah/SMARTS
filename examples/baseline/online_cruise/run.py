import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Silence the TF logs

import argparse
import pathlib
import warnings
from datetime import datetime
from typing import Any, Dict
from competition_env import CompetitionEnv

from ruamel.yaml import YAML
from cruise.env.make_env import make_env
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback

warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", category=ImportWarning)
warnings.simplefilter("ignore", category=ResourceWarning)
yaml = YAML(typ="safe")
import tensorflow as tf


def main(args: argparse.Namespace):
    # Load config file.
    config_file = yaml.load(
        (pathlib.Path(__file__).absolute().parent / "config.yaml").read_text()
    )

    # Load env config.
    config_env = config_file["smarts"]
    config_env["mode"] = args.mode
    config_env["headless"] = not args.head
    config_env["train_steps"] = args.train_steps
    config_env["scenarios_dir"] = (
        pathlib.Path(__file__).absolute().parents[0] / "scenarios/cruise"
    )
    _build_scenario()

    # Train or evaluate.
    if config_env["mode"] == "train" and not args.logdir:
        # Begin training from scratch.
        time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        logdir = pathlib.Path(__file__).absolute().parents[0] / "logs" / time
    elif (config_env["mode"] == "train" and args.logdir) or (
        config_env["mode"] == "evaluate"
    ):
        # Begin training from a pretrained model.
        logdir = pathlib.Path(args.logdir)
    else:
        raise KeyError(
            f'Expected \'train\' or \'evaluate\', but got {config_env["mode"]}.'
        )
    logdir.mkdir(parents=True, exist_ok=True)
    print("Logdir:", logdir)

    # Run training or evaluation.
    run(config_env, logdir)


def _build_scenario():
    scenario = str(pathlib.Path(__file__).absolute().parent / "scenarios/cruise")
    build_scenario = f"scl scenario build-all --clean {scenario}"
    os.system(build_scenario)


def run(config: Dict[str, Any], logdir: pathlib.PosixPath):
    env = make_env(config)

    if config["mode"] == "evaluate":
        print("Start evaluation.")
        model = PPO.load(logdir / "model.zip")
    elif config["mode"] == "train" and args.logdir:
        print("Start training from existing model.")
        model = PPO.load(logdir / "model.zip")
        model.set_env(env)
        model.learn(total_timesteps=config["train_steps"])
    else:
        print("Start training.")
        checkpoint_callback = CheckpointCallback(save_freq=50000, save_path=logdir,
                                        )
        model = PPO(
            "CnnPolicy",
            env,
            verbose=1,
            tensorboard_log=logdir / "tensorboard",
            use_sde=True,
            device='auto',
            batch_size=512
        )
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
        model.learn(total_timesteps=config["train_steps"], callback = checkpoint_callback)

    mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=config["eval_eps"], deterministic=True
    )
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    if config["mode"] == "train":
        model.save(logdir / "model")

    env.close()


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
    parser.add_argument(
        "--train-steps", help="Number of training steps.", type=int, default=1e6
    )

    args = parser.parse_args()

    if args.mode == "evaluate" and args.logdir is None:
        raise Exception("When --mode=evaluate, --logdir option must be specified.")

    main(args)
