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
                begin=("E5", i, 0),
                end=("E6", i, "max"),
            ),
            # Random flow rate, between x and y vehicles per minute.
            rate=60 * random.uniform(4, 5),
            # Random flow start time, between 0 and 10 seconds.
            begin=random.uniform(0, 10),
            # For an episode with maximum_episode_steps=3000 and step
            # time=0.1s, maximum episode time=300s. Hence, traffic set to
            # end at 900s, which is greater than maximum episode time of
            # 300s.
            end=60 * 15,            
            actors={freeway_car:1},
        )
        for i in range(3)
    # for _ in range(3)
    ]
)

route = Route(begin=("E8",0,1), end=("E6",0,'max'))
ego_missions = [
    Mission(
        route=route,
        start_time=15, # Delayed start, to ensure road has prior traffic.
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