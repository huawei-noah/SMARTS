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
# The author of this file is: https://github.com/mg2015started

from pathlib import Path

from smarts.sstudio.genscenario import gen_scenario
from smarts.sstudio.types import (
    Distribution,
    Flow,
    LaneChangingModel,
    Mission,
    RandomRoute,
    Route,
    Scenario,
    Traffic,
    TrafficActor,
)

social_vehicle_num = 100

ego_missions = [
    Mission(
        route=Route(begin=("edge-south-SN", 0, 10), end=("edge-east-WE", 0, 8)),
    ),
]

right_traffic_actor = TrafficActor(
    name="car",
    speed=Distribution(sigma=0.2, mean=1),
    lane_changing_model=LaneChangingModel(impatience=0),
)
scenario = Scenario(
    traffic={
        "basic": Traffic(
            flows=[
                Flow(
                    route=RandomRoute(),
                    rate=1,
                    actors={right_traffic_actor: 1.0},
                )
                for i in range(social_vehicle_num)
            ]
        )
    },
    ego_missions=ego_missions,
)

gen_scenario(scenario=scenario, output_dir=Path(__file__).parent)
