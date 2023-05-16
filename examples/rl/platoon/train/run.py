import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import sys
from pathlib import Path

# Required to load inference module
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import warnings
from datetime import datetime
from itertools import cycle
from typing import Any, Dict

import gym

# Load inference module to register agent
import inference
import stable_baselines3 as sb3lib
import torch as th
import yaml
from contrib_policy import network
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from train.env import make_env
from train.utils import ObjDict

from smarts.zoo import registry
from smarts.zoo.agent_spec import AgentSpec

# Disable GPU usage
th.cuda.is_available = lambda : False
device = th.device('cuda' if th.cuda.is_available() else 'cpu')

print("\n")
print(f"Torch cuda is available: {th.cuda.is_available()}")
print(f"Torch device: {device}")
print("\n")
warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", category=ResourceWarning)

# pytype: disable=attribute-error


def main(args: argparse.Namespace):
    # Load config file.
    parent_dir = Path(__file__).resolve().parent
    config_file = yaml.safe_load((parent_dir / "config.yaml").read_text())
    config = ObjDict(config_file["smarts"])

    # Load env config.
    config.mode = args.mode
    config.head = args.head

    # Setup logdir.
    if not args.logdir:
        time = datetime.now().strftime("%Y_%m_%d_%H_%M")
        logdir = parent_dir / "logs" / time
    else:
        logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    config.logdir = logdir
    print("\nLogdir:", logdir, "\n")

    # Setup model.
    if config.mode == "evaluate":
        # Begin evaluation.
        config.model = args.model
        print("\nModel:", config.model, "\n")
    elif config.mode == "train" and not args.model:
        # Begin training.
        pass
    else:
        raise KeyError(f"Expected 'train' or 'evaluate', but got {config.mode}.")

    # Make agent specification
    agent_spec = registry.make(locator=config.agent_locator)

    # Make training and evaluation environments.
    envs_train = {}
    envs_eval = {}
    for scenario in config.scenarios:
        scenario_path = str(Path(__file__).resolve().parents[4] / scenario)
        envs_train[f"{scenario}"] = make_env(
            env_id=config.env_id,
            scenario=scenario_path,
            agent_spec=agent_spec,
            config=config,
            seed=config.seed,
        )
        envs_eval[f"{scenario}"] = make_env(
            env_id=config.env_id,
            scenario=scenario_path,
            agent_spec=agent_spec,
            config=config,
            seed=config.seed,
        )

    # Run training or evaluation.
    run(
        envs_train=envs_train, envs_eval=envs_eval, config=config, agent_spec=agent_spec
    )

    print("Finished training ...")

    # Close all environments
    for env in envs_train.values():
        env.close()
    for env in envs_eval.values():
        env.close()


def run(
    envs_train: Dict[str, gym.Env],
    envs_eval: Dict[str, gym.Env],
    config: Dict[str, Any],
    agent_spec: AgentSpec,
):

    crop = agent_spec.agent_params["crop"]
    top_down_rgb = agent_spec.interface.top_down_rgb
    h = top_down_rgb.height - crop[2] - crop[3]
    w = top_down_rgb.width - crop[0] - crop[1]

    if config.mode == "train":
        print("\nStart training.\n")
        scenarios_iter = cycle(config.scenarios)
        model = sb3lib.PPO(
            env=envs_train[next(scenarios_iter)],
            tensorboard_log=config.logdir / "tensorboard",
            verbose=1,
            **network.combined_extractor(config),
        )

        # Print model summary
        # from torchinfo import summary
        # td = {"rgb":th.zeros(1,9,h,w)}
        # summary(model.policy, input_data=[td], depth=5)
        # input("Press any key to continue ...")

        for index in range(config.epochs):
            scen = next(scenarios_iter)
            env_train = envs_train[scen]
            env_eval = envs_eval[scen]
            print(f"\nTraining on {scen}.\n")
            checkpoint_callback = CheckpointCallback(
                save_freq=config.checkpoint_freq,
                save_path=config.logdir / "checkpoint",
                name_prefix=f"PPO",
            )
            eval_callback = EvalCallback(
                env_eval,
                best_model_save_path=config.logdir / "eval",
                n_eval_episodes=3,
                eval_freq=config.eval_freq,
                deterministic=True,
                render=False,
                verbose=1,
            )
            model.set_env(env_train)
            model.learn(
                total_timesteps=config.train_steps,
                callback=[checkpoint_callback, eval_callback],
                reset_num_timesteps=False,
            )

        # Save trained model.
        save_dir = config.logdir / "train"
        save_dir.mkdir(parents=True, exist_ok=True)
        time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        model.save(save_dir / ("model_" + time))
        print("\nSaved trained model.\n")

    if config.mode == "evaluate":
        print("\nEvaluate policy.\n")
        device = th.device("cpu")
        model = sb3lib.PPO.load(config.model, print_system_info=True, device=device)

        # Print model summary
        # from torchinfo import summary
        # td = {"rgb":th.zeros(1,9,h,w)}
        # summary(model.policy, input_data=[td], depth=5)
        # input("Press any key to continue ...")

        for env_name, env_eval in envs_eval.items():
            print(f"\nEvaluating env {env_name}.")
            mean_reward, std_reward = evaluate_policy(
                model, env_eval, n_eval_episodes=config.eval_eps, deterministic=True
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
    parser.add_argument(
        "--head", help="Display the simulation in Envision.", action="store_true"
    )

    args = parser.parse_args()

    if args.mode == "evaluate" and args.model is None:
        raise Exception("When --mode=evaluate, --model option must be specified.")

    main(args)
