import argparse
from pathlib import Path
from unittest import mock

from smarts.core.utils import import_utils

import_utils.import_module_from_file(
    "examples", Path(__file__).parents[1] / "__init__.py"
)


def _mock_load_config(load_config):
    def func():
        config = load_config()
        config.alg["n_steps"] = 20
        config.alg["batch_size"] = 5
        config.epochs = 2 * len(config.scenarios)
        config.train_steps = 40
        config.checkpoint_freq = 40
        config.eval_freq = 40
        config.eval_episodes = 1
        return config

    return func


def test_platoon():
    """Tests RL training of `examples/rl/platoon` example."""

    from examples.rl.platoon.train.run import load_config, main

    args = argparse.Namespace()
    args.mode = "train"
    args.logdir = None
    args.model = None
    args.head = False

    with mock.patch(
        "examples.rl.platoon.train.run.load_config",
        _mock_load_config(load_config),
    ):
        main(args)


def test_drive():
    """Tests RL training of `examples/rl/drive` example."""
    from examples.rl.drive.train.run import load_config, main

    args = argparse.Namespace()
    args.mode = "train"
    args.logdir = None
    args.model = None
    args.head = False

    with mock.patch(
        "examples.rl.drive.train.run.load_config",
        _mock_load_config(load_config),
    ):
        main(args)