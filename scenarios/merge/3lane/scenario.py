from pathlib import Path

import numpy as np
import random
from smarts.sstudio import gen_scenario
from smarts.sstudio.types import (
    Flow,
    Mission,
    Route,
    Scenario,
    Traffic,
    TrafficActor,
)
from smarts.sstudio.types import Distribution


freeway_car = TrafficActor(
    name="car",
    speed=Distribution(mean=0.5, sigma=0.8)
)

traffic = {}
traffic["merge"] = Traffic(
    flows=[
        Flow(
            route=Route(
                begin=("E5",random.randint(0,2), 0),
                end=("E6", random.randint(0,2), "max"),
            ),
            rate=3,
            begin=np.random.exponential(scale=2.5),
            actors={freeway_car:1},
        )
        for _ in range(3)
    for _ in range(3)
    ]
)

route = Route(begin=("E8",0,1), end=("E6",0,'max'))
ego_missions = [
    Mission(
        route=route,
        start_time=10, # Delayed start, to ensure road has prior traffic.
    )
]

gen_scenario(
    scenario=Scenario(
        traffic=traffic,
        ego_missions=ego_missions,
    ),
    output_dir=Path(__file__).parent,
)

# 1. exp distribution for for traffic, after 2 secs release one vehicle for lane0, 1 sec release on lane1...
# 2. uniformly select traffic density for each scenario
# 3. randomize the order of scenarios in env.