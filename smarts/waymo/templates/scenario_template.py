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

# Template for Waymo scenario.py

from pathlib import Path
import yaml
import os

from smarts.sstudio.genscenario import gen_scenario
from smarts.sstudio.types import Scenario, MapSpec

yaml_file = os.path.join(Path(__file__).parent, "waymo.yaml")
with open(yaml_file, "r") as yf:
    dataset_spec = yaml.safe_load(yf)["trajectory_dataset"]

dataset_path = dataset_spec["input_path"]
scenario_id = dataset_spec["scenario_id"]

gen_scenario(
    Scenario(
        map_spec=MapSpec(source=f"{dataset_path}#{scenario_id}", lanepoint_spacing=1.0),
        traffic_histories=["waymo.yaml"],
    ),
    output_dir=str(Path(__file__).parent),
    overwrite=True,
)
