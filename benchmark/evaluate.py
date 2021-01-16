import argparse
import ray
import os

from benchmark import gen_config
from benchmark.utils.rollout import rollout
from benchmark.metrics import basic_handler


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def parse_args():
    parser = argparse.ArgumentParser("Run evaluation")
    parser.add_argument(
        "scenario", type=str, help="Scenario name",
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--num_runs", type=int, default=10)
    # TODO(ming): eliminate this arg
    parser.add_argument(
        "--paradigm",
        type=str,
        default="decentralized",
        help="Algorithm paradigm, decentralized (default) or centralized",
    )
    parser.add_argument(
        "--headless", default=False, action="store_true", help="Turn on headless mode"
    )
    parser.add_argument("--config_file", "-f", type=str, required=True)
    parser.add_argument("--log_dir", type=str, default="./log/results")
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


def main(
    scenario,
    config_file,
    checkpoint,
    log_dir,
    num_steps=1000,
    num_episodes=10,
    paradigm="decentralized",
    headless=False,
    show_plots=False,
):

    config = gen_config(
        scenario=scenario,
        config_file=config_file,
        checkpoint=checkpoint,
        num_steps=num_steps,
        num_episodes=num_episodes,
        paradigm=paradigm,
        headless=headless,
        mode="evaluation",
    )

    ray.init()
    tune_config = config["run"]["config"]
    trainer_cls = config["trainer"]
    trainer_config = {"env_config": config["env_config"]}
    if paradigm != "centralized":
        trainer_config.update({"multiagent": tune_config["multiagent"]})
    else:
        trainer_config.update({"model": tune_config["model"]})

    trainer = trainer_cls(env=tune_config["env"], config=trainer_config)

    trainer.restore(checkpoint)
    metrics_handler = basic_handler.BasicMetricHandler(num_episodes)
    rollout(
        trainer, None, metrics_handler, num_steps, num_episodes, log_dir, show_plots
    )
    trainer.stop()


if __name__ == "__main__":
    args = parse_args()
    main(
        scenario=args.scenario,
        config_file=args.config_file,
        checkpoint=args.checkpoint,
        num_steps=args.num_steps,
        num_episodes=args.num_runs,
        paradigm=args.paradigm,
        headless=args.headless,
        show_plots=args.plot,
        log_dir=args.log_dir,
    )
