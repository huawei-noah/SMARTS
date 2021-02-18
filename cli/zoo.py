# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
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
import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

import click
from rich import print


@click.group(name="zoo")
def zoo_cli():
    pass


@zoo_cli.command(name="build", help="Build a policy")
@click.argument("policy", type=click.Path(exists=True), metavar="<policy>")
def build_policy(policy):
    def clean():
        subprocess.check_call([sys.executable, "setup.py", "clean", "--all"])

    def build():
        cwd = Path(os.getcwd())
        subprocess.check_call([sys.executable, "setup.py", "bdist_wheel"])
        results = sorted(glob.glob("./dist/*.whl"), key=os.path.getmtime, reverse=True)
        assert len(results) > 0, f"No policy package was built at path={cwd}"

        wheel = Path(results[0])
        dst_path = cwd / wheel.name
        shutil.move(wheel.resolve(), cwd / wheel.name)
        return dst_path

    os.chdir(policy)
    clean()
    wheel_path = build()
    clean()
    print(
        f"""
Policy built successfully and is available at,

\t[bold]{wheel_path}[/bold]

You can now add it to the policy zoo if you want to make it available to scenarios.
"""
    )


@zoo_cli.command(
    name="manager",
    help="Start the manager process which instantiates workers. Workers execute remote agents.",
)
@click.argument("port", default=7432, type=int)
def manager(port):
    from smarts.zoo import manager as zoo_manager

    zoo_manager.serve(port)


zoo_cli.add_command(build_policy)
zoo_cli.add_command(manager)
