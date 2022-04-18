import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf

import argparse
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import gym
import stable_baselines3 as sb3lib
import torch as th
from ruamel.yaml import YAML
from sb3 import action as sb3_action
from sb3 import info as sb3_info
from sb3 import observation as sb3_observation
from sb3 import policy as sb3_policy
from sb3 import reward as sb3_reward
from sb3 import util as sb3_util
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecMonitor

print("Torch cuda is available: ", th.cuda.is_available())
warnings.simplefilter("ignore", category=DeprecationWarning)
yaml = YAML(typ="safe")


def main(args: argparse.Namespace):
    # Load config file.
    config_file = yaml.load(
        (Path(__file__).absolute().parent / "config.yaml").read_text()
    )

    # Load env config.
    config = config_file["smarts"]
    config["head"] = args.head
    config["mode"] = args.mode

    # Setup logdir.
    print("\n\n")
    if (config["mode"] == "train" and args.model) or (config["mode"] == "evaluate"):
        # Begin training from a pretrained model.
        logdir = Path(args.logdir)
        config["model"] = args.model
        print("Model:", config["model"])
    elif config["mode"] == "train" and not args.model:
        # Begin training from scratch.
        time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        logdir = Path(__file__).absolute().parents[0] / "logs" / time
    else:
        raise KeyError(f'Expected \'train\' or \'evaluate\', but got {config["mode"]}.')
    logdir.mkdir(parents=True, exist_ok=True)
    config["logdir"] = logdir
    print("Logdir:", logdir)
    print("\n\n")

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
        headless=not config["head"],  # If False, enables Envision display.
        visdom=config["visdom"],  # If True, enables Visdom display.
        sumo_headless=not config["sumo_gui"],  # If False, enables sumo-gui display.
        img_meters=config["img_meters"],
        img_pixels=config["img_pixels"],
        action_space=config["action_space"],
    )

    # Wrap env with action, reward, and observation wrapper
    env = sb3_info.Info(env=env)
    env = sb3_action.Action(env=env, space=config["action_space"])
    env = sb3_reward.Reward(env=env)
    env = sb3_observation.Observation(env=env)

    # Check custom environment
    check_env(env)

    # Wrap env with SB3 wrappers
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(venv=env, n_stack=config["n_stack"], channels_order="first")
    env = VecMonitor(
        venv=env,
        filename=str(config["logdir"]),
        info_keywords=("is_success",),
    )

    # Record evaluation video
    # if not training:
    # env = VecVideoRecorder(
    #     venv=env,
    #     video_folder=str(config["logdir"] / "videos"),
    #     record_video_trigger=lambda x: x == 0,
    #     video_length=config["video_length"],
    #     name_prefix=config["name"]+"-PPO"
    # )

    return env


def run(env: gym.Env, eval_env: gym.Env, config: Dict[str, Any]):

    checkpoint_callback = CheckpointCallback(
        save_freq=config["checkpoint_freq"],
        save_path=config["logdir"] / "checkpoint",
        name_prefix=config["alg"],
    )
    eval_callback = EvalCallback(
        eval_env=eval_env,
        n_eval_episodes=config["eval_eps"],
        eval_freq=config["eval_freq"],
        log_path=config["logdir"] / "eval",
        best_model_save_path=config["logdir"] / "eval",
        deterministic=True,
    )
    # video_recorder_callback = sb3_callback.VideoRecorderCallback(
    #     env=eval_env,
    # )

    print("\n\n")
    if config["mode"] == "evaluate":
        print("Start evaluation.")
        model = getattr(sb3lib, config["alg"]).load(
            config["model"], print_system_info=True
        )
        sb3_util.print_model(model, eval_env)
    elif config["mode"] == "train" and config.get("model", None):
        print("Start training from existing model.")
        model = getattr(sb3lib, config["alg"]).load(
            config["model"], print_system_info=True
        )
        model.set_env(env)
        sb3_util.print_model(model, env)
        model.learn(
            total_timesteps=config["train_steps"],
            callback=[checkpoint_callback, eval_callback],
        )
    else:
        print("Start training from scratch.")
        model = getattr(sb3lib, config["alg"])(
            "CnnPolicy",
            env=env,
            verbose=1,
            tensorboard_log=config["logdir"] / "tensorboard",
            **(getattr(sb3_policy, config["policy"])(config)),
        )
        sb3_util.print_model(model, env)
        model.learn(
            total_timesteps=config["train_steps"],
            callback=[checkpoint_callback, eval_callback],
        )

    if config["mode"] == "train":
        save_dir = config["logdir"] / "train"
        save_dir.mkdir(parents=True, exist_ok=True)
        time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        model.save(save_dir / ("model_" + time))
        print("Saved trained model.")

    print("Evaluate policy.")
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=config["eval_eps"], deterministic=True
    )
    print(f"Mean reward:{mean_reward:.2f} +/- {std_reward:.2f}")
    print("Finished evaluating.")


if __name__ == "__main__":
    program = Path(__file__).stem
    parser = argparse.ArgumentParser(program)
    parser.add_argument(
        "--mode",
        help="`train` or `evaluate`. Default is `train`.",
        type=str,
        default="train",
    )
    parser.add_argument(
        "--logdir",
        help="Directory path for saving logs. Required if `--mode=evaluate`, else optional.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--model",
        help="Directory path to saved RL model. Required if `--mode=evaluate`, else optional.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--head", help="Run the simulation with display.", action="store_true"
    )

    args = parser.parse_args()

    if args.mode == "train" and args.model is not None and args.logdir is None:
        raise Exception(
            "When --mode=train, --model=<path>, --logdir option must be specified."
        )
    if args.mode == "evaluate" and (args.logdir is None or args.model is None):
        raise Exception(
            "When --mode=evaluate, --logdir and --model option must be specified."
        )

    main(args)


    # import torchvision.models as th_models
    # import torch
    # pip install prefetch_generator tqdm yacs


    # modelut = th_models.video.r2plus1d_18(pretrained=pretrained, progress=True)   
    # modelut = th.hub.load('datvuthanh/hybridnets', 'hybridnets', pretrained=True)
    # model = torch.hub.load('hustvl/yolop', 'yolop', pretrained=True)
    # Print model summary
    # print("\n\n")
    # network = sb3_util.Network(modelut)
    # print(model)
    # sb3_util.summary(model=model, depth=10, input_size=(1,3,256,256))
    # print("\n\n")