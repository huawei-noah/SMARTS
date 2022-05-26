import os
import warnings

import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from merge import agent as merge_agent
from merge import buffer as merge_buffer
from merge import env as merge_env
from merge import network as merge_network
from ruamel.yaml import YAML
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.eval.metric_utils import log_metrics
from tf_agents.metrics import tf_metrics
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.utils.common import Checkpointer, function

warnings.simplefilter("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.filterwarnings(
    "ignore",
    ".*Box.*",
)
logging.getLogger().setLevel(logging.INFO)
# Suppress tensorflow deprecation warning messages
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

yaml = YAML(typ="safe")

print(f"\nTF version: {tf.version.VERSION}\n")

tf.random.set_seed(42)
physical_devices = tf.config.list_physical_devices("GPU")
print("Physical devices:", physical_devices)
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


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


def run(train_env, eval_env, config: Dict[str, Any]):
    if config["mode"] == "evaluate":
        print("\nStart evaluation.\n")
        eval_step = tf.Variable(0, dtype=tf.int64)
        eval_summary_writer = tf.summary.create_file_writer(
            logdir=str(config["logdir"] / "tensorboard" / "eval")
        )
        # Restore checkpoint
        # train_checkpointer.initialize_or_restore()
        evaluate(eval_env, policy, eval_step, eval_summary_writer, config)
    elif config["mode"] == "train" and config.get("model", None):
        print("\nStart training from an existing model.\n")
        # model = getattr(sb3lib, config["alg"]).load(
        #     config["model"], print_system_info=True
        # )
        # model.set_env(env)
        # merge_util.print_model(model, env, config["alg"])
        # model.learn(
        #     total_timesteps=config["train_steps"],
        #     callback=[checkpoint_callback, eval_callback],
        # )
    else:
        print("\nStart training from scratch.\n")
        train(train_env, eval_env, config)

    # if config["mode"] == "train":
    #     save_dir = config["logdir"] / "train"
    #     save_dir.mkdir(parents=True, exist_ok=True)
    #     time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    #     model.save(save_dir / ("model_" + time))
    #     print("\nSaved trained model.\n")

    # print("\nEvaluate policy.\n")
    # mean_reward, std_reward = evaluate_policy(
    #     model, eval_env, n_eval_episodes=config["eval_eps"], deterministic=True
    # )
    # print(f"Mean reward:{mean_reward:.2f} +/- {std_reward:.2f}")
    # print("\nFinished evaluating.\n")


def train(train_env, eval_env, config):
    # Build agent, network, and replay buffer
    network = getattr(merge_network, config["network"])(env=train_env)
    agent = getattr(merge_agent, config["agent"])(
        env=train_env, network=network, config=config
    )
    agent.train_step_counter.assign(0)
    replay_buffer, replay_buffer_observer = getattr(merge_buffer, config["buffer"])(
        env=train_env, agent=agent, config=config
    )

    # Train metrics
    train_metrics = [
        tf_metrics.NumberOfEpisodes(),
        tf_metrics.EnvironmentSteps(),
        tf_metrics.AverageReturnMetric(batch_size=train_env.batch_size),
        tf_metrics.AverageEpisodeLengthMetric(batch_size=train_env.batch_size),
        tf_metrics.MaxReturnMetric(),
    ]

    # Build the driver, and dataset
    collect_driver = DynamicStepDriver(
        env=train_env,
        policy=agent.collect_policy,
        observers=[replay_buffer_observer] + train_metrics,
        num_steps=config["driver"][
            "num_steps"
        ],  # collect `num_steps` steps for each iteration
    )
    # Dataset generates trajectories with shape [BxTx...] where
    # B = `sample_batch_size` and T = `num_steps` = `n_step_update` + 1. Here,
    # `n_step_update` is the number of steps to consider when computing TD
    # error and TD loss in DqnAgent.
    dataset = replay_buffer.as_dataset(
        sample_batch_size=config["dataset"]["batch_size"],
        num_steps=config["agent_kwargs"]["n_step_update"] + 1,
        num_parallel_calls=3,
    ).prefetch(3)
    iterator = iter(dataset)

    assert (
        config["driver"]["num_steps"]
        >= (config["agent_kwargs"]["n_step_update"] + 1)
        * config["dataset"]["batch_size"]
    ), "Driver collects lesser steps than that required for training per iteration."

    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    # collect_driver.run = function(collect_driver.run)
    # agent.train = function(agent.train)

    # Create checkpoint
    train_checkpointer = Checkpointer(
        ckpt_dir=config["logdir"] / "checkpoint",
        max_to_keep=3,
        agent=agent,
        # replay_buffer=replay_buffer,
        global_step=agent.train_step_counter,
        # metrics=metric_utils.MetricsGroup(train_metrics, 'train_metrics')
    )

    # Setup Tensorboard
    train_summary_writer = tf.summary.create_file_writer(
        logdir=str(config["logdir"] / "tensorboard" / "train")
    )
    eval_summary_writer = tf.summary.create_file_writer(
        logdir=str(config["logdir"] / "tensorboard" / "eval")
    )

    # Start training
    for _ in range(config["train"]["iterations"]):
        # Collect a few steps using collect_policy and save to the replay buffer.
        collect_driver.run()

        # Sample a batch of data from the buffer and update the agent's network.
        trajectories, buffer_info = next(iterator)
        train_loss = agent.train(trajectories)

        train_step = agent.train_step_counter.numpy()
        if train_step % config["train"]["checkpoint_interval"] == 0:
            train_checkpointer.save(gloabl_step=train_step)
        if train_step % config["log_interval"] == 0:
            with train_summary_writer.as_default():
                tf.summary.scalar(
                    name="reward", data=train_loss.result(), step=train_step
                )
        if train_step % config["eval"]["interval"] == 0:
            print("Evaluating. Step = {train_step}.")
            evaluate(eval_env, agent.policy, train_step, eval_summary_writer, config)

    return


def evaluate(env, policy, step, summary_writer, config):
    total_return = 0.0
    total_step = 0
    for ep in range(config["eval"]["episodes"]):
        time_step = env.reset()
        ep_return = 0.0
        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = env.step(action_step.action)
            ep_return += time_step.reward
            total_step += 1

        print(f"Episode {ep} return: {ep_return.numpy()}")
        total_return += ep_return
        with summary_writer.as_default():
            tf.summary.scalar(
                name="episode return", data=ep_return.numpy(), step=step + total_step
            )

    avg_return = total_return / config["eval"]["episodes"]
    print(f"Average episode return: {avg_return.numpy()}")

    return avg_return.numpy()


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
