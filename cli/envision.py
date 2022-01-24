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
import click

from envision.server import run


@click.group(name="envision")
def envision_cli():
    pass


@envision_cli.command(name="start")
@click.option("-p", "--port", help="Port Envision will run on.", default=8081)
@click.option(
    "-s",
    "--scenarios",
    help="A list of directories where scenarios are stored.",
    multiple=True,
    default=["scenarios"],
)
@click.option(
    "-c",
    "--max_capacity",
    help=(
        "Max capacity in MB of Envision's playback buffer. The larger the more contiguous history "
        "Envision can store."
    ),
    default=500,
    type=float,
)
def start_server(port, scenarios, max_capacity):
    run(scenario_dirs=scenarios, max_capacity_mb=max_capacity, port=port)


envision_cli.add_command(start_server)
