from pathlib import Path
from unittest import mock
import argparse

from smarts.core.utils import import_utils


import_utils.import_module_from_file(
    "examples", Path(__file__).parents[1] / "__init__.py"
)


def _mock_load_config(load_config):
    def func():
        config = load_config()
        config.alg["n_steps"] = 20
        config.alg["batch_size"] = 5
        config.epochs = 2*len(config.scenarios)
        config.train_steps = 40
        config.checkpoint_freq = 40
        config.eval_freq = 40
        config.eval_episodes = 1
        return config

    return func


def test_platoon():
    from examples.rl.platoon.train.run import main, load_config

    args = argparse.Namespace()
    args.mode = 'train'
    args.logdir = None
    args.model = None
    args.head = False

    with mock.patch(
        "examples.rl.platoon.train.run.load_config",
        _mock_load_config(load_config),
    ):
        main(args)


def test_drive():
    from examples.rl.drive.train.run import main, load_config

    args = argparse.Namespace()
    args.mode = 'train'
    args.logdir = None
    args.model = None
    args.head = False

    with mock.patch(
        "examples.rl.drive.train.run.load_config",
        _mock_load_config(load_config),
    ):
        main(args)
