"""
Example for centralized critic training for homogeneous agents.
"""

import argparse
from pathlib import Path

import tensorflow as tf
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.fcnet_v2 import FullyConnectedNetwork
from ray.tune.schedulers import PopulationBasedTraining
from tensorflow.keras import layers

# from smarts.core.utils.file import copy_tree
from smarts.env.rllib_hiway_env import RLlibHiWayEnv
from smarts.trainer.utils import LayerConfig, pack_custom_options_with_opponent
from smarts.trainer.framework import dect
from smarts.trainer.zoo import MA_POLICIES
from smarts.trainer.zoo.utils.utils import explore
from smarts.trainer.zoo.tuned import SPACES
from smarts.trainer.zoo.tuned.callback import (
    on_episode_start,
    on_episode_step,
    on_episode_end,
)

RUN_NAME = Path(__file__).stem
EXPERIMENT_NAME_FMT = "{scenario}-{algorithm}-{n_agent}"


def parse_args():
    parser = argparse.ArgumentParser("CC-Training")

    # env setting
    parser.add_argument(
        "--scenario", required=True, type=str, help="Path to a scenario"
    )
    parser.add_argument("--num_agents", type=int, default=2, help="Agent number")
    parser.add_argument(
        "--headless", default=False, action="store_true", help="Turn on headless mode"
    )
    parser.add_argument(
        "--space_type",
        default="simple",
        choices=set(SPACES.keys()),
        help="Space types of trainable agents (defalut is simple).",
    )

    # training setting
    parser.add_argument(
        "--policy",
        type=str,
        default="centralized_a2c",
        help="Multi-agent policy",
        choices=set(MA_POLICIES.keys()),
    )
    parser.add_argument("--num_workers", type=int, default=4, help="rllib num workers")
    parser.add_argument(
        "--horizon", type=int, default=1000, help="horizon for a episode"
    )
    parser.add_argument(
        "--resume", default=False, action="store_true", help="Resume training or not."
    )
    parser.add_argument(
        "--debug", default=False, action="store_true", help="Debug smarts or not"
    )
    # not work in multi trial
    parser.add_argument(
        "--restore",
        default=None,
        type=str,
        help="path to restore checkpoint, absolute dir",
    )
    parser.add_argument(
        "--log_dir",
        default="results",
        type=str,
        help="path to store checkpoint model, relative dir",
    )

    parser.add_argument(
        "--num_gpus", type=int, default=0, help="Set number of gpus, default is 0"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of times to sample from hyperparameter space",
    )
    parser.add_argument(
        "--eager", default=False, action="store_true", help="turn on eager mode."
    )

    return parser.parse_args()


def main(args):
    space = SPACES[args.space_type]
    META = MA_POLICIES[args.policy].META
    # agent config
    agent_ids = {f"AGENT-{i}" for i in range(args.num_agents)}
    agent_specs = {
        agent_id: space.rllib_trainable_agent()["agent_spec"] for agent_id in agent_ids
    }

    # !!!IMPORTANT!!! we must retrieve the embedding spaces manually.
    obs_pre = ModelCatalog.get_preprocessor_for_space(space.OBSERVATION_SPACE)
    action_pre = ModelCatalog.get_preprocessor_for_space(space.ACTION_SPACE)
    network_config = {
        "actor": FullyConnectedNetwork,
        "critic": [  # net configuration for centralized critic
            LayerConfig(
                layers.Dense, {"units": 16, "activation": tf.nn.tanh, "name": "c_l1"}
            ),
            LayerConfig(layers.Dense, {"units": 1, "activation": None, "name": "c_l2"}),
        ],
    }

    custom_policy_options = {
        agent_id: pack_custom_options_with_opponent(
            agent_id,
            opponent_ids=agent_ids - {agent_id},
            options=network_config,
            oppo_obs_pre=obs_pre,
            oppo_action_pre=action_pre,
        )
        for agent_id in agent_ids
    }

    scenario_path = Path(args.scenario).absolute()
    # ====================================
    # init log and checkpoint dir_info
    # ====================================
    experiment_name = EXPERIMENT_NAME_FMT.format(
        scenario=scenario_path.name,
        algorithm=META.policy.__name__,
        n_agent=args.num_agents,
    )

    log_dir = Path(args.log_dir).expanduser().absolute() / RUN_NAME
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpointing at {log_dir}")

    if args.restore:
        restore_path = Path(args.restore).expanduser()
        print(f"Loading model from {restore_path}")
    else:
        restore_path = None

    pbt = PopulationBasedTraining(
        time_attr="time_total_s",
        metric="episode_reward_mean",
        mode="max",
        perturbation_interval=300,
        resample_probability=0.25,
        hyperparam_mutations={"lr": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]},
        custom_explore_fn=explore,
    )

    policies = {
        agent_id: (
            None,
            space.OBSERVATION_SPACE,
            space.ACTION_SPACE,
            {
                "model": {
                    "custom_model": dect.FRAMEWORK_NAME,
                    "custom_options": custom_policy_options[agent_id],
                },
            },
        )
        for agent_id in agent_ids
    }

    # XXX: We turned off model saving and restore temporary since there
    # is an issue that cannot save custom models in multi-agent cases.
    # Fortunately, we found the solution and it will be fixed by refactoring RLlib components.
    analysis = tune.run(
        META.trainer,
        name=experiment_name,
        stop={"time_total_s": 5 * 60 * 60},
        checkpoint_freq=1,
        checkpoint_at_end=True,
        local_dir=str(log_dir),
        resume=args.resume,
        restore=restore_path,
        max_failures=1000,
        num_samples=args.num_samples,
        export_formats=["model", "checkpoint"],
        config={
            "env": RLlibHiWayEnv,
            "log_level": "ERROR",
            "batch_mode": "complete_episodes",
            "eager": args.eager,
            "env_config": {  # required by env.__init__
                "seed": 42,
                "scenarios": [str(scenario_path)],
                "headless": args.headless,
                "agent_specs": agent_specs,
            },
            "num_workers": args.num_workers,
            "num_gpus": args.num_gpus,
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": lambda agent_id: agent_id,
            },
            "callbacks": {
                "on_episode_start": on_episode_start,
                "on_episode_step": on_episode_step,
                "on_episode_end": on_episode_end,
            },
        },
        scheduler=pbt,
    )

    print(analysis.dataframe().head())

    # TODO: Currently, we cannot export models because RLLib has issues with models named
    #       something different from "default_model".

    # max_reward_logdir = analysis.get_best_logdir("episode_reward_max")
    # model_path = Path(max_reward_logdir) / "model"
    # dest_model_path = log_dir / "model"

    # copy_tree(model_path, dest_model_path, overwrite=True)
    # print(f"Wrote model to: {dest_model_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
