# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
import os

import click


@click.group(
    name="waymo",
    help="Utilities for using the Waymo Motion Dataset with SMARTS. See `scl waymo COMMAND --help` for further options.",
)
def waymo_cli():
    pass


@waymo_cli.command(
    name="overview", help="Display summary info for each scenario in the TFRecord file."
)
@click.argument(
    "tfrecord_file", type=click.Path(exists=True), metavar="<tfrecord_file>"
)
def overview(tfrecord_file: str):
    from tabulate import tabulate

    from smarts.waymo import waymo_utils

    scenarios = waymo_utils.get_tfrecord_info(tfrecord_file)
    rows = [
        [k, v["timestamps"], v["vehicles"], v["pedestrians"]]
        for k, v in scenarios.items()
    ]
    print(
        tabulate(
            rows,
            headers=[
                "Scenario ID",
                "Timestamps",
                "Vehicles",
                "Pedestrians",
            ],
        )
    )


@waymo_cli.command(
    name="preview", help="Plot the map and trajectories of the scenario."
)
@click.argument(
    "tfrecord_file", type=click.Path(exists=True), metavar="<tfrecord_file>"
)
@click.argument("scenario_id", type=str, metavar="<scenario_id>")
@click.option(
    "--animate",
    is_flag=True,
    default=False,
    help="Animate the vehicle trajectories.",
)
@click.option(
    "--label_vehicles",
    is_flag=True,
    default=False,
    help="Plot the initial positions of all vehicles with their IDs.",
)
def preview(
    tfrecord_file: str,
    scenario_id: str,
    animate: bool,
    label_vehicles: bool,
):
    from smarts.waymo import waymo_utils

    waymo_utils.plot_scenario(tfrecord_file, scenario_id, animate, label_vehicles)


@waymo_cli.command(
    name="export", help="Export the Waymo scenario to a SMARTS scenario."
)
@click.argument(
    "tfrecord_file", type=click.Path(exists=True), metavar="<tfrecord_file>"
)
@click.argument("scenario_id", type=str, metavar="<scenario_id>")
@click.argument(
    "export_folder", type=click.Path(exists=False), metavar="<export_folder>"
)
def export(
    tfrecord_file: str,
    scenario_id: str,
    export_folder: str,
):
    from smarts.waymo import waymo_utils

    scenario_folder = os.path.join(export_folder, scenario_id)
    if not os.path.exists(scenario_folder):
        os.makedirs(scenario_folder)
    scenario_file = os.path.join(scenario_folder, "scenario.py")
    with open(scenario_file, "w") as f:
        f.write(waymo_utils.gen_smarts_scenario_code(tfrecord_file, scenario_id))


waymo_cli.add_command(overview)
waymo_cli.add_command(preview)
waymo_cli.add_command(export)
