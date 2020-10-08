import click

from .envision import envision_cli
from .studio import scenario_cli
from .zoo import zoo_cli


@click.group()
def scl():
    pass


scl.add_command(envision_cli)
scl.add_command(scenario_cli)
scl.add_command(zoo_cli)


if __name__ == "__main__":
    scl()
