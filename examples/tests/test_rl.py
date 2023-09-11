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
        config.alg["n_steps"] = 4
        config.alg["batch_size"] = 2
        config.alg["n_epochs"] = 2
        config.epochs = 2 * len(config.scenarios)
        config.train_steps = 8
        config.checkpoint_freq = 10000
        config.eval_freq = 10000
        return config

    return func


def test_e11_platoon():
    """Tests RL training of `examples/e11_platoon` example."""

    from examples.e11_platoon.train.run import load_config, main

    args = argparse.Namespace()
    args.mode = "train"
    args.logdir = None
    args.model = None
    args.head = False

    with mock.patch(
        "examples.e11_platoon.train.run.load_config",
        _mock_load_config(load_config),
    ):
        main(args)


def test_e10_drive():
    """Tests RL training of `examples/e10_drive` example."""
    from examples.e10_drive.train.run import load_config, main

    args = argparse.Namespace()
    args.mode = "train"
    args.logdir = None
    args.model = None
    args.head = False

    with mock.patch(
        "examples.e10_drive.train.run.load_config",
        _mock_load_config(load_config),
    ):
        main(args)
