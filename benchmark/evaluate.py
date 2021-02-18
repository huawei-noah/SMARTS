# MIT License
#
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
import argparse
import os

import ray

from benchmark import gen_config
from benchmark.metrics import basic_handler
from benchmark.utils.rollout import rollout

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def parse_args():
    parser = argparse.ArgumentParser("Run evaluation")
    parser.add_argument(
        "scenario",
        type=str,
        help="Scenario name",
    )
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
    parser.add_argument("--config_files", "-f", type=str, nargs="+", required=True)
    parser.add_argument("--log_dir", type=str, default="./log/results")
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


def main(
    scenario,
    config_files,
    log_dir,
    num_steps=1000,
    num_episodes=10,
    paradigm="decentralized",
    headless=False,
    show_plots=False,
):

    ray.init()
    metrics_handler = basic_handler.BasicMetricHandler()

    for config_file in config_files:
        config = gen_config(
            scenario=scenario,
            config_file=config_file,
            num_steps=num_steps,
            num_episodes=num_episodes,
            paradigm=paradigm,
            headless=headless,
            mode="evaluation",
        )

        tune_config = config["run"]["config"]
        trainer_cls = config["trainer"]
        trainer_config = {"env_config": config["env_config"]}
        if paradigm != "centralized":
            trainer_config.update({"multiagent": tune_config["multiagent"]})
        else:
            trainer_config.update({"model": tune_config["model"]})

        trainer = trainer_cls(env=tune_config["env"], config=trainer_config)

        trainer.restore(config["checkpoint"])
        metrics_handler.set_log(
            algorithm=config_file.split("/")[-2], num_episodes=num_episodes
        )
        rollout(trainer, None, metrics_handler, num_steps, num_episodes, log_dir)
        trainer.stop()

    if show_plots:
        metrics_handler.show_plots()


if __name__ == "__main__":
    args = parse_args()
    main(
        scenario=args.scenario,
        config_files=args.config_files,
        num_steps=args.num_steps,
        num_episodes=args.num_runs,
        paradigm=args.paradigm,
        headless=args.headless,
        show_plots=args.plot,
        log_dir=args.log_dir,
    )
