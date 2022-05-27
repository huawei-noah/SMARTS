import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

tf.get_logger().setLevel("ERROR")  # To avoid TF warnings

import warnings

warnings.simplefilter("ignore", category=DeprecationWarning)

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from merge import agent as merge_agent
from merge import buffer as merge_buffer
from merge import env as merge_env
from merge import network as merge_network
from ruamel.yaml import YAML
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
from tf_agents.metrics import tf_metrics
from tf_agents.policies.random_tf_policy import RandomTFPolicy
from tf_agents.utils.common import Checkpointer, function

yaml = YAML(typ="safe")

print(f"\nTF version: {tf.version.VERSION}\n")
tf.random.set_seed(42)
print(f"\nPhysical devices: {tf.config.list_physical_devices()} \n")
gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


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
        config["checkpoint_dir"] = args.model
        print("\nCheckpoint directory:", config["checkpoint_dir"], "\n")
    elif config["mode"] == "train" and not args.model:
        # Begin training from scratch.
        config["checkpoint_dir"] = config["logdir"] / "checkpoint"
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

    # Build agent, and network
    network = getattr(merge_network, config["network"])(env=train_env)
    agent = getattr(merge_agent, config["agent"])(
        env=train_env, network=network, config=config
    )
    agent.train_step_counter.assign(0)

    # Create or restore checkpoint
    train_checkpointer = Checkpointer(
        ckpt_dir=config["checkpoint_dir"],
        max_to_keep=1,
        agent=agent,
    )
    train_checkpointer.initialize_or_restore()

    # Train, re-train, or evaluate
    if config["mode"] == "evaluate":
        print("\nStart evaluation.\n")
        eval_step = tf.Variable(0, dtype=tf.int64)
        summary_writer = tf.summary.create_file_writer(
            logdir=str(config["logdir"] / "tensorboard")
        )
        evaluate(
            env=eval_env,
            policy=agent.policy,
            step=eval_step,
            summary_writer=summary_writer,
            config=config,
        )
    elif config["mode"] == "train" and config.get("model", None):
        print("\nStart training from an existing model.\n")
        train(train_env, eval_env, agent, train_checkpointer, config)
    else:
        print("\nStart training from scratch.\n")
        train(train_env, eval_env, agent, train_checkpointer, config)


def train(train_env, eval_env, agent, train_checkpointer, config):

    # Build replay buffer
    replay_buffer, replay_buffer_observer = getattr(merge_buffer, config["buffer"])(
        env=train_env, agent=agent, config=config
    )

    # Train metrics
    train_metrics = [
        tf_metrics.EnvironmentSteps(),
        tf_metrics.NumberOfEpisodes(),
        tf_metrics.AverageReturnMetric(batch_size=train_env.batch_size),
        tf_metrics.AverageEpisodeLengthMetric(batch_size=train_env.batch_size),
        tf_metrics.MaxReturnMetric(),
    ]

    # Build the driver, and dataset
    initial_collect_policy = RandomTFPolicy(
        train_env.time_step_spec(), train_env.action_spec()
    )
    initial_driver = DynamicStepDriver(
        env=train_env,
        policy=initial_collect_policy,
        observers=[replay_buffer_observer],
        num_steps=config["driver"][
            "initial_steps"
        ],  # collect `initial_steps` steps to prefill the buffer with random policy
    )
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

    # assert (
    #     config["driver"]["num_steps"]
    #     >= (config["agent_kwargs"]["n_step_update"] + 1)
    #     * config["dataset"]["batch_size"]
    # ), "Driver collects lesser steps than that required for training per iteration."

    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    # collect_driver.run = function(collect_driver.run)
    # agent.train = function(agent.train)

    # Setup tensorboard
    summary_writer = tf.summary.create_file_writer(
        logdir=str(config["logdir"] / "tensorboard")
    )

    # Start training
    initial_driver.run()
    train_step = agent.train_step_counter.numpy()
    env_step = train_metrics[0].result()
    for _ in range(config["train_iterations"]):
        if train_step % 1e3 == 0:
            print(f"Training. Train_step = {train_step}. Env_step = {env_step}.")

        # Collect a few steps using collect_policy and save to the replay buffer.
        collect_driver.run()
        train_step = agent.train_step_counter.numpy()
        env_step = train_metrics[0].result()

        # Sample a batch of data from the buffer and update the agent's network.
        trajectories, buffer_info = next(iterator)
        train_loss = agent.train(trajectories)

        if train_step % config["checkpoint_interval"] == 0:
            train_checkpointer.save(global_step=train_step)
        if train_step % config["log_interval"] == 0:
            with summary_writer.as_default():
                tf.summary.scalar(
                    name="train/loss",
                    data=train_loss.loss.numpy(),
                    step=env_step,
                )
                for train_metric in train_metrics:
                    train_metric.tf_summaries(step_metrics=train_metrics[:1])
        if train_step % config["eval"]["interval"] == 0:
            print(f"Evaluating. Train_step = {train_step}. Env_step = {env_step}.")
            evaluate(
                env=eval_env,
                policy=agent.policy,
                step=env_step,
                summary_writer=summary_writer,
                config=config,
            )

    # Finally, save a checkpoint and evaluate the final model.
    train_checkpointer.save(global_step=train_step)
    print(f"Evaluating. Train_step = {train_step}. Env_step = {env_step}.")
    evaluate(
        env=eval_env,
        policy=agent.policy,
        step=env_step,
        summary_writer=summary_writer,
        config=config,
    )

    return


def evaluate(env, policy, step, summary_writer, config):
    total_return = 0.0
    for _ in range(config["eval"]["episodes"]):
        time_step = env.reset()
        ep_return = 0.0
        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = env.step(action_step.action)
            ep_return += time_step.reward

        # print(f"Eval episode {ep} return: {ep_return.numpy()[0]:.2f}")
        total_return += ep_return

    avg_return = total_return / config["eval"]["episodes"]
    with summary_writer.as_default():
        tf.summary.scalar(
            name="eval/episode avg return", data=avg_return.numpy()[0], step=step
        )
    print(f"Evaluating. Episode average return: {avg_return.numpy()[0]:.2f}")

    return


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
