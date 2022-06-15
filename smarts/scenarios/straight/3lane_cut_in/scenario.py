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

import random
from pathlib import Path

import numpy as np

from smarts.sstudio import gen_scenario
from smarts.sstudio.types import (
    Distribution,
    Flow,
    Mission,
    Route,
    Scenario,
    Traffic,
    TrafficActor,
)

normal = TrafficActor(
    name="car",
    speed=Distribution(mean=0.5, sigma=0.8),
)

traffic = Traffic(
    flows=[
        Flow(
            route=Route(
                begin=("E1", random.randint(0, 2), "random"),
                end=("E2", random.randint(0, 2), "max"),
            ),
            rate=3,
            begin=np.random.exponential(scale=2.5),
            actors={normal: 1},
        )
        for _ in range(3)
        for _ in range(4)
    ]
)
ego_mission = [Mission(Route(begin=("E1", 1, 1), end=("E2", 1, "max")))]

scenario = Scenario(
    traffic={"all": traffic},
    ego_missions=ego_mission,
)

gen_scenario(scenario, output_dir=str(Path(__file__).parent))
