import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import argparse
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import gym
import stable_baselines3 as sb3lib
import tensorflow as tf
import torch as th
from ruamel.yaml import YAML
from intersection import action as intersection_action
from intersection import info as intersection_info
from intersection import observation as intersection_observation
from intersection import policy as intersection_policy
from intersection import reward as intersection_reward
from intersection import util as intersection_util
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecMonitor

print("\nTorch cuda is available: ", th.cuda.is_available(), "\n")
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
    if not args.logdir:
        time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        logdir = Path(__file__).absolute().parents[0] / "logs" / time
    else:
        logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    config["logdir"] = logdir
    print("\nLogdir:", logdir, "\n")

    # Setup model.
    if (config["mode"] == "train" and args.model) or (config["mode"] == "evaluate"):
        # Begin training or evaluation from a pretrained model.
        config["model"] = args.model
        print("\nModel:", config["model"], "\n")
    elif config["mode"] == "train" and not args.model:
        # Begin training from scratch.
        pass
    else:
        raise KeyError(f'Expected \'train\' or \'evaluate\', but got {config["mode"]}.')

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
    )

    # Wrap env with action, reward, and observation wrapper
    env = intersection_info.Info(env=env)
    env = intersection_action.Action(env=env, space=config["action_wrapper"])
    env = intersection_reward.Reward(env=env)
    env = getattr(intersection_observation, config["observation_wrapper"])(env=env)

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

    if config["mode"] == "evaluate":
        print("\nStart evaluation.\n")
        model = getattr(sb3lib, config["alg"]).load(
            config["model"], print_system_info=True
        )
        intersection_util.print_model(model, env, config["alg"])
    elif config["mode"] == "train" and config.get("model", None):
        print("\nStart training from existing model.\n")
        model = getattr(sb3lib, config["alg"]).load(
            config["model"], print_system_info=True
        )
        model.set_env(env)
        intersection_util.print_model(model, env, config["alg"])
        model.learn(
            total_timesteps=config["train_steps"],
            callback=[checkpoint_callback, eval_callback],
        )
    else:
        print("\nStart training from scratch.\n")
        model = getattr(sb3lib, config["alg"])(
            env=env,
            verbose=1,
            tensorboard_log=config["logdir"] / "tensorboard",
            **(getattr(intersection_policy, config["policy"])(config)),
        )
        intersection_util.print_model(model, env, config["alg"])
        model.learn(
            total_timesteps=config["train_steps"],
            callback=[checkpoint_callback, eval_callback],
        )

    if config["mode"] == "train":
        save_dir = config["logdir"] / "train"
        save_dir.mkdir(parents=True, exist_ok=True)
        time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        model.save(save_dir / ("model_" + time))
        print("\nSaved trained model.\n")

    print("\nEvaluate policy.\n")
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=config["eval_eps"], deterministic=True
    )
    print(f"Mean reward:{mean_reward:.2f} +/- {std_reward:.2f}")
    print("\nFinished evaluating.\n")


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

    if args.mode == "evaluate" and args.model is None:
        raise Exception("When --mode=evaluate, --model option must be specified.")

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

    # import gym
    # import Box2D
    # env = gym.make('RoadRunner-v0')
    # print("obs_space",env.observation_space)
    # print("action_space",env.action_space)

    # import gym
    # import torch as th

    # from stable_baselines3 import PPO

    # # Custom actor (pi) and value function (vf) networks
    # # of two layers of size 32 each with Relu activation function
    # policy_kwargs = dict(activation_fn=th.nn.ReLU,
    #                     net_arch=[dict(pi=[32, 32], vf=[32, 32])])
    # # Create the agent
    # model = PPO("MlpPolicy", "CartPole-v1", policy_kwargs=policy_kwargs, verbose=1)
    # # Retrieve the environment
    # env = model.get_env()
    # print(env.observation_space)
