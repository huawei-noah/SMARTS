"""This file demonstrates how it might be possible to use `click` and `hydra` together."""
from dataclasses import dataclass

import click
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf


@dataclass
class Hand:
    strength: str = 0


@dataclass
class Top:
    hand: Hand


@dataclass
class BaseCfg:
    name: str
    top: Top


cs = ConfigStore.instance()
cs.store("base_cfg", node=BaseCfg)


@click.command()
@click.argument("settings", nargs=-1, type=click.Path())
def display_config(settings):
    """This command consumes command line arguments and forwards them to the ``hydra compose``
    interface."""
    with hydra.initialize("clicktest_dir", version_base=None):
        cfg = hydra.compose("base.yaml", overrides=settings)
        typed_cfg = OmegaConf.to_object(cfg)
        print(typed_cfg)


@click.group()
def experiment_configuration():
    """
    A configuration entry.
    Use --help with each command for further information.
    """
    pass


experiment_configuration.add_command(display_config)

if __name__ == "__main__":
    experiment_configuration()
