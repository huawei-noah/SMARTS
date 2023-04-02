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

from smarts.sstudio.genscenario import gen_scenario
from smarts.sstudio.types import (
    Flow,
    MapZone,
    Mission,
    Route,
    Scenario,
    Traffic,
    TrafficActor,
    TrapEntryTactic,
)

intersection_car = TrafficActor(
    name="car",
)

vertical_routes = [
    ("north-NS", "south-NS"),
    ("south-SN", "north-SN"),
]

horizontal_routes = [
    ("west-WE", "east-WE"),
    ("east-EW", "west-EW"),
]

turn_left_routes = [
    ("south-SN", "west-EW"),
    ("west-WE", "north-SN"),
    ("north-NS", "east-WE"),
    ("east-EW", "south-NS"),
]

turn_right_routes = [
    ("south-SN", "east-WE"),
    ("west-WE", "south-NS"),
    ("north-NS", "west-EW"),
    ("east-EW", "north-SN"),
]

# Total route combinations = 12C1 + 12C2 + 12C3 + 12C4 = 793
all_routes = vertical_routes + horizontal_routes + turn_left_routes + turn_right_routes
route_comb = [com for elems in range(1, 5) for com in combinations(all_routes, elems)]
traffic = {}
for name, routes in enumerate(route_comb):
    traffic[str(name)] = Traffic(
        flows=[
            Flow(
                route=Route(
                    begin=(f"edge-{r[0]}", 0, 0),
                    end=(f"edge-{r[1]}", 0, "max"),
                ),
                # Random flow rate, between 3 and 5 vehicles per minute.
                rate=60 * random.uniform(3, 5),
                # Random flow start time, between 0 and 10 seconds.
                begin=random.uniform(0, 10),
                # For an episode with maximum_episode_steps=3000 and step
                # time=0.1s, the maximum episode time=300s. Hence, traffic is
                # set to end at 900s, which is greater than maximum episode
                # time of 300s.
                end=60 * 15,
                actors={intersection_car: 1},
            )
            for r in routes
        ]
    )

route = Route(begin=("edge-west-WE", 0, 60), end=("edge-north-SN", 0, 40))
ego_missions = [
    Mission(
        route=route,
        start_time=12,  # Delayed start, to ensure road has prior traffic.
        entry_tactic=TrapEntryTactic(
            wait_to_hijack_limit_s=1,
            zone=MapZone(
                start=(
                    route.begin[0],
                    route.begin[1],
                    route.begin[2] - 5,
                ),
                length=10,
                n_lanes=1,
            ),
            default_entry_speed=5,
        ),
    ),
]

scnr_path = Path(__file__).parent
gen_scenario(
    scenario=Scenario(
        traffic=traffic,
        ego_missions=ego_missions,
    ),
    output_dir=scnr_path,
)
