import os
import warnings

import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import gym
from gym.utils.env_checker import check_env
from gym.wrappers.frame_stack import FrameStack
from merge import action as merge_action
from merge import agent as merge_agent
from merge import network as merge_network
from merge import observation as merge_observation
from merge import reward as merge_reward

# from merge import policy as merge_policy
# from merge import util as merge_util
from ruamel.yaml import YAML
from tf_agents.environments import (
    PyEnvironment,
    suite_gym,
    tf_environment,
    tf_py_environment,
    validate_py_environment,
)

warnings.simplefilter("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.filterwarnings(
    "ignore",
    ".*Box.*",
)

yaml = YAML(typ="safe")

print(f"\nTF version: {tf.version.VERSION}\n")


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
        # Begin training or evaluation from a pre-trained agent.
        config["model"] = args.model
        print("\nModel:", config["model"], "\n")
    elif config["mode"] == "train" and not args.model:
        # Begin training from scratch.
        pass
    else:
        raise KeyError(f'Expected \'train\' or \'evaluate\', but got {config["mode"]}.')

    # Make training and evaluation environments.
    train_env = make_env(config=config)
    eval_env = make_env(config=config)

    # Run training or evaluation.
    run(train_env=train_env, eval_env=eval_env, config=config)
    train_env.close()
    eval_env.close()


def make_env(config: Dict[str, Any]) -> PyEnvironment:
    # Create environment in Gym.
    # Refer to https://www.gymlibrary.ml/
    # env = gym.make(
    #     "smarts.env:merge-v0",
    #     headless=not config["head"],  # If False, enables Envision display.
    #     visdom=config["visdom"],  # If True, enables Visdom display.
    #     sumo_headless=not config["sumo_gui"],  # If False, enables sumo-gui display.
    #     img_meters=config["img_meters"],
    #     img_pixels=config["img_pixels"],
    # )
    # print(env.action_space)
    # env = merge_reward.Reward(env=env)
    # env = merge_action.Action(env=env, space=config["action_wrapper"])
    # env = getattr(merge_observation, config["observation_wrapper"])(env=env)
    # check_env(env)

    # Create the equivalent environment in TF.
    # Refer to https://www.tensorflow.org/agents/tutorials/2_environments_tutorial
    gym_reward_wrapper = lambda env: merge_reward.Reward(env=env)
    gym_action_wrapper = lambda env: merge_action.Action(
        env=env, space=config["action_wrapper"]
    )
    gym_obs_wrapper = lambda env: getattr(
        merge_observation, config["observation_wrapper"]
    )(env=env)
    gym_frame_stack = lambda env: FrameStack(env=env, num_stack=config["num_stack"])
    pyenv = suite_gym.load(
        environment_name="smarts.env:merge-v0",
        gym_env_wrappers=[
            gym_reward_wrapper,
            gym_action_wrapper,
            gym_obs_wrapper,
            gym_frame_stack,
        ],
        gym_kwargs={
            "headless": not config["head"],  # If False, enables Envision display.
            "visdom": config["visdom"],  # If True, enables Visdom display.
            "sumo_headless": not config[
                "sumo_gui"
            ],  # If False, enables sumo-gui display.
            "img_meters": config["img_meters"],
            "img_pixels": config["img_pixels"],
        },
    )
    validate_py_environment(environment=pyenv)
    # (Optional) Manually verify Py env spaces
    # print('action_spec:', pyenv.action_spec())
    # print('time_step_spec.observation:', pyenv.time_step_spec().observation)
    # print('time_step_spec.step_type:', pyenv.time_step_spec().step_type)
    # print('time_step_spec.discount:', pyenv.time_step_spec().discount)
    # print('time_step_spec.reward:', pyenv.time_step_spec().reward)

    tfenv = tf_py_environment.TFPyEnvironment(pyenv)
    # (Optional) Manually verify TF env specs
    # print(isinstance(tfenv, tf_environment.TFEnvironment))
    # print("TimeStep Specs:", tfenv.time_step_spec())
    # print("Action Specs:", tfenv.action_spec())

    return tfenv


def run(train_env: gym.Env, eval_env: gym.Env, config: Dict[str, Any]):

    # checkpoint_callback = CheckpointCallback(
    #     save_freq=config["checkpoint_freq"],
    #     save_path=config["logdir"] / "checkpoint",
    #     name_prefix=config["alg"],
    # )
    # eval_callback = EvalCallback(
    #     eval_env=eval_env,
    #     n_eval_episodes=config["eval_eps"],
    #     eval_freq=config["eval_freq"],
    #     log_path=config["logdir"] / "eval",
    #     best_model_save_path=config["logdir"] / "eval",
    #     deterministic=True,
    # )

    network = getattr(merge_network, config["network"])(env=env)
    agent = getattr(merge_agent, config["agent"])(env=env)

    return

    if config["mode"] == "evaluate":
        print("\nStart evaluation.\n")
        model = getattr(sb3lib, config["alg"]).load(
            config["model"], print_system_info=True
        )
        merge_util.print_model(model, env, config["alg"])
    elif config["mode"] == "train" and config.get("model", None):
        print("\nStart training from an existing model.\n")
        model = getattr(sb3lib, config["alg"]).load(
            config["model"], print_system_info=True
        )
        model.set_env(env)
        merge_util.print_model(model, env, config["alg"])
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
            **(getattr(merge_policy, config["policy"])(config)),
        )
        merge_util.print_model(model, env, config["alg"])
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
