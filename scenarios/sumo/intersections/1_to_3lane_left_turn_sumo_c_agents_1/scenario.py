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
from itertools import combinations
from pathlib import Path

from smarts.sstudio import gen_scenario
from smarts.sstudio.sstypes import (
    Flow,
    Mission,
    Route,
    Scenario,
    ScenarioMetadata,
    Traffic,
    TrafficActor,
)

normal = TrafficActor(
    name="car",
)

vertical_routes = [
    ("E2", 0, "E7", 0),
    ("E8", 0, "E1", 1),
]

horizontal_routes = [
    ("E3", 0, "E5", 0),
    ("E3", 1, "E5", 1),
    ("E3", 2, "E5", 2),
    ("E6", 1, "E4", 1),
    ("E6", 0, "E4", 0),
]

turn_left_routes = [
    ("E8", 0, "E5", 2),
    ("E6", 1, "E1", 1),
    ("E2", 1, "E4", 1),
    ("E3", 2, "E7", 0),
]

turn_right_routes = [
    ("E6", 0, "E7", 0),
    ("E3", 0, "E1", 0),
    ("E2", 0, "E5", 0),
    ("E8", 0, "E4", 0),
]

# Total route combinations
all_routes = vertical_routes + horizontal_routes + turn_left_routes + turn_right_routes
route_comb = [com for elems in range(4, 5) for com in combinations(all_routes, elems)]
traffic = {}
for name, routes in enumerate(route_comb):
    traffic[str(name)] = Traffic(
        flows=[
            Flow(
                route=Route(
                    begin=(f"{r[0]}", r[1], 0),
                    end=(f"{r[2]}", r[3], "max"),
                ),
                # Random flow rate, between x and y vehicles per minute.
                rate=60 * random.uniform(2, 5),
                # Random flow start time, between x and y seconds.
                begin=random.uniform(0, 3),
                # For an episode with maximum_episode_steps=3000 and step
                # time=0.1s, maximum episode time=300s. Hence, traffic set to
                # end at 900s, which is greater than maximum episode time of
                # 300s.
                end=60 * 15,
                actors={normal: 1},
            )
            for r in routes
        ]
    )

route = Route(begin=("E8", 0, 5), end=("E5", 0, "max"))

default_speed = 13
route_length = 100
duration = (route_length / default_speed) * 2

ego_missions = [
    Mission(
        route=route,
        start_time=15,  # Delayed start, to ensure road has prior traffic.
    )
]

gen_scenario(
    scenario=Scenario(
        traffic=traffic,
        ego_missions=ego_missions,
        scenario_metadata=ScenarioMetadata(
            scenario_difficulty=0.6,
            scenario_duration=duration,
        ),
    ),
    output_dir=Path(__file__).parent,
)
