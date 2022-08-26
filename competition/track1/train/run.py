import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import argparse
import warnings
from datetime import datetime
from itertools import cycle
from pathlib import Path
from typing import Any, Dict

import gym
import stable_baselines3 as sb3lib
import torch as th
from ruamel.yaml import YAML
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from train import env as multi_scenario_env
import network

print("\nTorch cuda is available: ", th.cuda.is_available(), "\n")
warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", category=ResourceWarning)
yaml = YAML(typ="safe")


def main(args: argparse.Namespace):
    # Load config file.
    config_file = yaml.load(
        (Path(__file__).absolute().parent / "config.yaml").read_text()
    )

    # Load env config.
    config = config_file["smarts"]
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
    if config["mode"] == "evaluate":
        # Begin evaluation.
        config["model"] = args.model
        print("\nModel:", config["model"], "\n")
    elif config["mode"] == "train" and not args.model:
        # Begin training.
        pass
    else:
        raise KeyError(f'Expected \'train\' or \'evaluate\', but got {config["mode"]}.')

    # Make training and evaluation environments.
    envs_train = {}
    envs_eval = {}
    wrappers = multi_scenario_env.wrappers(config=config)
    for scen in config["scenarios"]:
        envs_train[f"{scen}"] = multi_scenario_env.make(
            config=config, scenario=scen, wrappers=wrappers
        )
        envs_eval[f"{scen}"] = multi_scenario_env.make(
            config=config, scenario=scen, wrappers=wrappers
        )

    # Run training or evaluation.
    run(envs_train=envs_train, envs_eval=envs_eval, config=config)

    # Close all environments
    for env in envs_train.values():
        env.close()
    for env in envs_eval.values():
        env.close()


def run(
    envs_train: Dict[str, gym.Env],
    envs_eval: Dict[str, gym.Env],
    config: Dict[str, Any],
):

    if config["mode"] == "train":
        print("\nStart training.\n")
        scenarios_iter = cycle(config["scenarios"])
        model = getattr(sb3lib, config["alg"])(
            env=envs_train[next(scenarios_iter)],
            verbose=1,
            tensorboard_log=config["logdir"] / "tensorboard",
            **network.combined_extractor(config),
        )
        for index in range(config["epochs"]):
            scen = next(scenarios_iter)
            env_train = envs_train[scen]
            env_eval = envs_eval[scen]
            print(f"\nTraining on {scen}.\n")
            checkpoint_callback = CheckpointCallback(
                save_freq=config["checkpoint_freq"],
                save_path=config["logdir"] / "checkpoint",
                name_prefix=f"{config['alg']}_{index}",
            )
            model.set_env(env_train)
            model.learn(
                total_timesteps=config["train_steps"],
                callback=[checkpoint_callback],
            )

        # Save trained model.
        save_dir = config["logdir"] / "train"
        save_dir.mkdir(parents=True, exist_ok=True)
        time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        model.save(save_dir / ("model_" + time))
        print("\nSaved trained model.\n")

    if config["mode"] == "evaluate":
        print("\nEvaluate policy.\n")
        model = getattr(sb3lib, config["alg"]).load(
            config["model"], print_system_info=True
        )
        for env_name, env_eval in envs_eval.items():
            print(f"\nEvaluating env {env_name}.")
            mean_reward, std_reward = evaluate_policy(
                model, env_eval, n_eval_episodes=config["eval_eps"], deterministic=True
            )
            print(f"Mean reward:{mean_reward:.2f} +/- {std_reward:.2f}\n")
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
        help="Directory path for saving logs.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--model",
        help="Directory path to saved RL model. Required if `--mode=evaluate`.",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    if args.mode == "evaluate" and args.model is None:
        raise Exception("When --mode=evaluate, --model option must be specified.")

    main(args)
