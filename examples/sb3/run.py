import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Silence the TF logs

import argparse
import pathlib
import warnings
from datetime import datetime
from typing import Any, Dict

import gym
from sb3.common import action as common_action
from sb3.common import observation as common_observation
from sb3.common import reward as common_reward
from ruamel.yaml import YAML
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy

warnings.simplefilter("ignore", category=DeprecationWarning)
yaml = YAML(typ="safe")


def main(args: argparse.Namespace):
    # Load config file.
    config_file = yaml.load(
        (pathlib.Path(__file__).absolute().parent / "config.yaml").read_text()
    )

    # Load env config.
    config_env = config_file["smarts"]
    config_env["mode"] = args.mode
    config_env["train_steps"] = args.train_steps

    # Create environment
    env = gym.make(
        "smarts.env:intersection-v0",
        headless=not args.head,  # If False, enables Envision display.
        visdom=config_env["visdom"],  # If True, enables Visdom display.
        sumo_headless=not config_env["sumo_gui"],  # If False, enables sumo-gui display.
    )
    # Wrap env with action, reward, and observation wrapper
    env = common_action.Action(env=env)
    env = common_reward.Reward(env=env)
    env = common_observation.Observation(env=env)
    # Check our custom environment compatibility with SB3
    check_env(env)

    # Setup train or evaluate.
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
    run(env, config_env, logdir)
    env.close()


def run(env: gym.Env, config: Dict[str, Any], logdir: pathlib.Path):

    if config["mode"] == "evaluate":
        print("Start evaluation.")
        model = PPO.load(logdir / "model.zip")
    elif config["mode"] == "train" and args.logdir:
        print("Start training from existing model.")
        model = PPO.load(logdir / "model.zip")
        model.set_env(env)
        model.learn(total_timesteps=config["train_steps"])
    else:
        print("Start training from scratch.")
        model = PPO(
            "CnnPolicy",
            env,
            verbose=1,
            tensorboard_log=logdir / "tensorboard",
            use_sde=True,
        )
        model.learn(total_timesteps=config["train_steps"])

    print("Evaluate policy")
    mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=config["eval_eps"], deterministic=True
    )
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    if config["mode"] == "train":
        model.save(logdir / "model")


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
        "--train-steps", help="Number of training steps.", type=int, default=5e6
    )

    args = parser.parse_args()

    if args.mode == "evaluate" and args.logdir is None:
        raise Exception("When --mode=evaluate, --logdir option must be specified.")

    main(args)
