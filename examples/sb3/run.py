import os

import tensorflow

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Silence the TF logs

import argparse
import pathlib
import warnings
from datetime import datetime
from typing import Any, Dict

import torch as th
import gym
from ruamel.yaml import YAML
from sb3 import action as sb3_action
from sb3 import observation as sb3_observation
from sb3 import reward as sb3_reward
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecMonitor, VecNormalize

warnings.simplefilter("ignore", category=DeprecationWarning)
yaml = YAML(typ="safe")


def main(args: argparse.Namespace):
    # Load config file.
    config_file = yaml.load(
        (pathlib.Path(__file__).absolute().parent / "config.yaml").read_text()
    )

    # Load env config.
    config = config_file["smarts"]
    config["mode"] = args.mode

    # Setup logdir.
    if config["mode"] == "train" and not args.logdir:
        # Begin training from scratch.
        time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        logdir = pathlib.Path(__file__).absolute().parents[0] / "logs" / time
    elif (config["mode"] == "train" and args.logdir) or (
        config["mode"] == "evaluate"
    ):
        # Begin training from a pretrained model.
        logdir = pathlib.Path(args.logdir)
    else:
        raise KeyError(
            f'Expected \'train\' or \'evaluate\', but got {config["mode"]}.'
        )
    logdir.mkdir(parents=True, exist_ok=True)
    config["logdir"] = logdir
    print("Logdir:", logdir)

    # Make training and evaluation environments.
    env = make_env(config=config, training=True)
    eval_env = make_env(config=config, training=False)

    # Run training or evaluation.
    run(env=env, eval_env=eval_env, config=config)
    env.close()


def make_env(config: Dict[str, Any], training: bool) -> gym.Env:
    # Create environment
    env = gym.make(
        "smarts.env:intersection-v0",
        headless=not args.head,  # If False, enables Envision display.
        visdom=config["visdom"],  # If True, enables Visdom display.
        sumo_headless=not config["sumo_gui"],  # If False, enables sumo-gui display.
    )

    # Wrap env with action, reward, and observation wrapper
    env = sb3_action.Action(env=env)
    env = sb3_reward.Reward(env=env)
    env = sb3_observation.Observation(env=env, n_stack=1)

    # Check custom environment
    check_env(env)

    #  Wrap env with SB3 wrappers
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(venv=env, n_stack=config["n_stack"], channels_order="last")
    #Not needed# env = VecNormalize(venv=env, training=training)
    env = VecMonitor(venv=env, filename=str(config["logdir"]))

    return env


def run(env: gym.Env, eval_env: gym.Env, config: Dict[str, Any]):

    checkpoint_callback = CheckpointCallback(
        save_freq=config["checkpoint_freq"],
        save_path=config["logdir"],
        name_prefix="model",
    )

    if config["mode"] == "evaluate":
        print("Start evaluation.")
        model = PPO.load(config["logdir"] / "model.zip")
    elif config["mode"] == "train" and args.logdir:
        print("Start training from existing model.")
        model = PPO.load(config["logdir"] / "model.zip")
        model.set_env(env)
        model.learn(total_timesteps=config["train_steps"], callback=checkpoint_callback)
    else:
        print("Start training from scratch.")
        policy_kwargs = dict(
            # activation_fn=th.nn.ReLU,
            # activation_fn=th.nn.Tanh, # default activation used
            net_arch=[[128], dict(pi=[32, 32], vf=[32, 32])]
        )
        # Default CNN policy used
        # self.cnn = nn.Sequential(
        #     nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
        #     nn.ReLU(),
        #     nn.Flatten(),
        # )
        model = PPO(
            "CnnPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=config["logdir"] / "tensorboard",
            seed = config["seed"]
        )

        model.learn(total_timesteps=config["train_steps"], callback=checkpoint_callback)
        # eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/',
        # log_path='./logs/', eval_freq=500,
        # deterministic=True, render=False)

    if config["mode"] == "train":
        model.save(config["logdir"] / "model")
        print("Saved trained model.")

    print("Evaluate policy.")
    # ------------------------------------------------
    # set deterministic=True when calling the .predict() in evaluating PPO
    # ------------------------------------------------
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=config["eval_eps"], deterministic=True)
    print(f"Mean reward:{mean_reward:.2f} +/- {std_reward:.2f}")
    print("Finished evaluating.")


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

    args = parser.parse_args()

    if args.mode == "evaluate" and args.logdir is None:
        raise Exception("When --mode=evaluate, --logdir option must be specified.")

    main(args)
