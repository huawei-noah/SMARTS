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
import os
import time
import sys
import signal
import subprocess
import webbrowser
from contextlib import contextmanager

import click


@contextmanager
def kill_process_group_afterwards():
    os.setpgrp()
    try:
        yield
    finally:
        # Kill all processes in my group
        os.killpg(0, signal.SIGKILL)


@click.command(
    name="run",
    help="Run an experiment on a scenario",
    context_settings=dict(ignore_unknown_options=True),
)
@click.option(
    "--envision",
    is_flag=True,
    default=False,
    help="Start up Envision server at the specified port when running an experiment",
)
@click.option(
    "-p",
    "--envision_port",
    help="Port on which Envision will run.",
    default=None,
)
@click.argument(
    "script_path", type=click.Path(exists=True), metavar="<script>", required=True
)
@click.argument("script_args", nargs=-1, type=click.UNPROCESSED)
def run_experiment(envision, envision_port, script_path, script_args):
    with kill_process_group_afterwards():
        if envision:
            if envision_port is None:
                envision_port = 8081
            subprocess.Popen(
                [
                    "scl",
                    "envision",
                    "start",
                    "-s",
                    "./scenarios",
                    "-p",
                    str(envision_port),
                ],
            )
            # Just in case: give Envision a bit of time to warm up
            time.sleep(2)
            url = "http://localhost:" + str(envision_port)
            webbrowser.open_new_tab(url)

        if (not envision) and envision_port:
            click.echo(
                "Port passed without starting up the envision server. Use the --envision option to start the server along with the --envision port option."
            )

        script = subprocess.Popen(
            [sys.executable, script_path, *script_args],
        )
        script.communicate()
