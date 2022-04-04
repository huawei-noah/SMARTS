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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import click

from cli.envision import envision_cli
from cli.run import run_experiment
from cli.studio import scenario_cli
from cli.ultra import ultra_cli
from cli.zoo import zoo_cli
from pathlib import Path

@click.group()
def scl():
    """
    The SMARTS command line interface.
    Use --help with each command for further information.
    """
    pass

def recursive_help(cmd, file, parent=None):
    ctx = click.core.Context(cmd, info_name=cmd.name, parent=parent)
    line = "=" * len(cmd.name)
    file.write(line + "\n" + cmd.name + "\n" + line + "\n\n")
    file.write(cmd.get_help(ctx) + "\n\n")
    commands = getattr(cmd, 'commands', {})
    for sub in commands.values():
        recursive_help(sub, file, ctx)

@scl.command(name="document-help",help="Write SCL help information to docs.")
def document():
    directory = str(Path(__file__).parent.parent)
    f = open(directory + "/docs/sim/cli.rst", "w")
    recursive_help(scl, f)
    f.close

scl.add_command(envision_cli)
scl.add_command(scenario_cli)
scl.add_command(ultra_cli)
scl.add_command(zoo_cli)
scl.add_command(run_experiment)
scl.add_command(document)

if __name__ == "__main__":
    scl()