import os
import sys
import glob
import shutil
import subprocess
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


zoo_cli.add_command(build_policy)
