# MIT License

# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import click

from cli.benchmark import benchmark_cli
from cli.diagnostic import diagnostic_cli
from cli.envision import envision_cli
from cli.run import run_experiment
from cli.studio import scenario_cli
from cli.zoo import zoo_cli


@click.group()
def scl():
    """
    The SMARTS command line interface.
    Use --help with each command for further information.
    """
    pass


scl.add_command(envision_cli)
scl.add_command(benchmark_cli)
scl.add_command(scenario_cli)
scl.add_command(zoo_cli)
scl.add_command(run_experiment)

try:
    from cli.waymo import waymo_cli
except (ModuleNotFoundError, ImportError):

    @click.group(
        name="waymo",
        invoke_without_command=True,
        help="The `scl waymo` command requires `[waymo]`.",
    )
    @click.pass_context
    def waymo_cli(ctx):
        click.echo(
            "The `scl waymo` command is unavailable. To enable, pip install the missing dependencies.\n"
            "pip install pathos==0.2.8 tabulate>=0.8.10 waymo-open-dataset-tf-2-4-0"
        )


scl.add_command(waymo_cli)
scl.add_command(diagnostic_cli)

if __name__ == "__main__":
    scl()
