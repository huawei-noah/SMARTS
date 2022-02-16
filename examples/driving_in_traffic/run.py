# This reinforcement learning example uses code from DreamerV2 (https://github.com/danijar/dreamerv2).

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Silence the TF logs

import argparse
import collections
import logging
import pathlib
import re
import warnings
from datetime import datetime
from typing import Callable, Generator

import gym
import numpy as np
import rich.traceback
import tensorflow as tf
from driving_in_traffic import seed
from driving_in_traffic.env import single_agent
from ruamel.yaml import YAML

warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", category=PendingDeprecationWarning)
import dreamerv2 as dv2  # isort: skip
import dreamerv2.api as api  # isort:skip
import dreamerv2.agent as agent  # isort:skip
import dreamerv2.common as common  # isort:skip

logging.getLogger().setLevel("ERROR")
warnings.filterwarnings("ignore", ".*box bound precision lowered.*")
rich.traceback.install()
yaml = YAML(typ="safe")
seed(42)


def main(args: argparse.Namespace):
    # Load config file.
    config_file = yaml.load(
        (pathlib.Path(__file__).absolute().parent / "config.yaml").read_text()
    )

    # Load SMARTS env config.
    config_env = config_file["smarts"]
    config_env["mode"] = args.mode
    config_env["headless"] = not args.head
    config_env["scenarios_dir"] = (
        pathlib.Path(__file__).absolute().parents[0] / "scenarios"
    )
    _build_scenario()

    # Load dreamerv2 config.
    config_dv2 = dv2.api.defaults
    config_dv2 = config_dv2.update(config_file["dreamerv2"])

    # Setup tensorflow.
    tf.config.run_functions_eagerly(not config_dv2.jit)

    # Setup GPU.
    _setup_gpu()

    # Train or evaluate.
    if config_env["mode"] == "train" and not args.logdir:
        # Train from scratch.
        time = datetime.now().strftime("%Y_%m_%d_%H_%M")
        logdir = pathlib.Path(__file__).absolute().parents[0] / "logs" / time
    elif config_env["mode"] == "train" and args.logdir:
        # Begin training from a pretrained model.
        logdir = args.logdir
    elif config_env["mode"] == "evaluate":
        logdir = args.logdir
        config_dv2 = config_dv2.update({"eval_eps": 1e8})
    else:
        raise KeyError(
            f'Expected \'train\' or \'evaluate\', but got {config_env["mode"]}.'
        )
    config_dv2 = config_dv2.update({"logdir": logdir})

    # Create SMARTS env.
    gen_env = single_agent.gen_env(config_env, config_env["seed"])

    # Run training or evaluation.
    run(config_dv2, gen_env, config_env["mode"])


def _build_scenario():
    scenario = str(pathlib.Path(__file__).absolute().parent / "scenarios")
    build_scenario = f"scl scenario build-all --clean {scenario}"
    os.system(build_scenario)


def _setup_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)
    else:
        warnings.warn(
            f"Not configured to use GPU or GPU not available.",
            ResourceWarning,
        )


def _wrap_env(env: gym.Env, config: common.config.Config):
    env = dv2.common.GymWrapper(env)
    env = dv2.common.ResizeImage(env)
    if hasattr(env.act_space["action"], "n"):
        env = dv2.common.OneHotAction(env)
    else:
        env = dv2.common.NormalizeAction(env)
    env = dv2.common.TimeLimit(env, config.time_limit)
    return env


def run(
    config: common.config.Config,
    gen_env: Generator[Callable[[str], gym.Env], None, None],
    mode: str,
):
    logdir = pathlib.Path(config.logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)
    config.save(logdir / "config.yaml")
    print(config, "\n")
    print("Logdir:", logdir)

    train_replay = dv2.common.Replay(logdir / "train_episodes", **config.replay)
    eval_replay = dv2.common.Replay(
        logdir / "eval_episodes",
        **dict(
            capacity=config.replay.capacity // 10,
            minlen=1,
            maxlen=config.dataset.length,
        ),
    )
    step = dv2.common.Counter(train_replay.stats["total_steps"])
    outputs = [
        dv2.common.TerminalOutput(),
        dv2.common.TensorBoardOutput(logdir),
    ]
    logger = dv2.common.Logger(step, outputs, multiplier=config.action_repeat)
    metrics = collections.defaultdict(list)

    should_train = dv2.common.Every(config.train_every)
    should_log = dv2.common.Every(config.log_every)
    should_video_train = dv2.common.Every(config.eval_every)
    should_video_eval = dv2.common.Every(config.eval_every)
    should_expl = dv2.common.Until(config.expl_until)

    def per_episode(ep, mode):
        length = len(ep["reward"]) - 1
        score = float(ep["reward"].astype(np.float64).sum())
        print(f"{mode.title()} episode has {length} steps and return {score:.1f}.")
        logger.scalar(f"{mode}_return", score)
        logger.scalar(f"{mode}_length", length)
        for key, value in ep.items():
            if re.match(config.log_keys_sum, key):
                logger.scalar(f"sum_{mode}_{key}", ep[key].sum())
            if re.match(config.log_keys_mean, key):
                logger.scalar(f"mean_{mode}_{key}", ep[key].mean())
            if re.match(config.log_keys_max, key):
                logger.scalar(f"max_{mode}_{key}", ep[key].max(0).mean())
        should = {"train": should_video_train, "eval": should_video_eval}[mode]
        if should(step):
            for key in config.log_keys_video:
                logger.video(f"{mode}_policy_{key}", ep[key])
        replay = dict(train=train_replay, eval=eval_replay)[mode]
        logger.add(replay.stats, prefix=mode)
        logger.write()

    print("Create envs.")
    train_envs = []
    if mode == "train":
        train_envs = [_wrap_env(next(gen_env)(env_name="train"), config)]
        train_driver = dv2.common.Driver(train_envs)
        train_driver.on_episode(lambda ep: per_episode(ep, mode="train"))
        train_driver.on_step(lambda tran, worker: step.increment())
        train_driver.on_step(train_replay.add_step)
        train_driver.on_reset(train_replay.add_step)
    eval_envs = [_wrap_env(next(gen_env)(env_name="eval"), config)]
    act_space = eval_envs[0].act_space
    obs_space = eval_envs[0].obs_space
    eval_driver = dv2.common.Driver(eval_envs)
    eval_driver.on_episode(lambda ep: per_episode(ep, mode="eval"))
    eval_driver.on_episode(eval_replay.add_episode)

    prefill = max(0, config.prefill - train_replay.stats["total_steps"])
    if prefill and mode == "train":
        print(f"Prefill dataset ({prefill} steps).")
        random_agent = dv2.common.RandomAgent(act_space)
        train_driver(random_agent, steps=prefill, episodes=1)
        eval_driver(random_agent, episodes=1)
        train_driver.reset()
        eval_driver.reset()

    print("Create agent.")
    train_dataset = iter(train_replay.dataset(**config.dataset))
    report_dataset = iter(train_replay.dataset(**config.dataset))
    eval_dataset = iter(eval_replay.dataset(**config.dataset))
    agnt = dv2.agent.Agent(config, obs_space, act_space, step)
    train_agent = dv2.common.CarryOverState(agnt.train)
    train_agent(next(train_dataset))
    if (logdir / "variables.pkl").exists():
        agnt.load(logdir / "variables.pkl")
    else:
        print("Pretrain agent.")
        for _ in range(config.pretrain):
            train_agent(next(train_dataset))
    train_policy = lambda *args: agnt.policy(
        *args, mode="explore" if should_expl(step) else "train"
    )
    eval_policy = lambda *args: agnt.policy(*args, mode="eval")

    def train_step(tran, worker):
        if should_train(step):
            for _ in range(config.train_steps):
                mets = train_agent(next(train_dataset))
                [metrics[key].append(value) for key, value in mets.items()]
        if should_log(step):
            for name, values in metrics.items():
                logger.scalar(name, np.array(values, np.float64).mean())
                metrics[name].clear()
            logger.add(agnt.report(next(report_dataset)), prefix="train")
            logger.write(fps=True)

    if mode == "train":
        train_driver.on_step(train_step)

    while step < config.steps:
        logger.write()
        print("Start evaluation.")
        logger.add(agnt.report(next(eval_dataset)), prefix="eval")
        eval_driver(eval_policy, episodes=config.eval_eps)
        if mode == "train":
            print("Start training.")
            train_driver(train_policy, steps=config.eval_every)
            agnt.save(logdir / "variables.pkl")
    for env in train_envs + eval_envs:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser("driving_in_traffic")
    parser.add_argument(
        "--mode",
        help="`train` or `evauate`. Default is `train`.",
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
