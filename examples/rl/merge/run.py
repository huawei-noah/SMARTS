import os
import warnings

import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import gym
from merge import agent as merge_agent
from merge import buffer as merge_buffer
from merge import env as merge_env
from merge import network as merge_network
from ruamel.yaml import YAML

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
    train_env = merge_env.make(config=config)
    eval_env = merge_env.make(config=config)

    # Run training or evaluation.
    run(train_env=train_env, eval_env=eval_env, config=config)
    train_env.close()
    eval_env.close()


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

    network = getattr(merge_network, config["network"])(env=train_env)
    agent = getattr(merge_agent, config["agent"])(
        env=train_env, network=network, config=config
    )
    replay_buffer, replay_buffer_observer = getattr(merge_buffer, config["buffer"])(
        env=train_env, agent=agent, config=config
    )

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
