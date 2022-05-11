from pathlib import Path

import numpy as np
import random
from smarts.sstudio import gen_scenario
from smarts.sstudio import types as t
from smarts.sstudio.types import Distribution

traffic = t.Traffic(
    flows=[
        t.Flow(
            route=t.Route(
                begin=("E5",random.randint(0,2), 'random'),
                end=("E6", random.randint(0,2), "max"),
            ),
            rate=3,
            begin=np.random.exponential(scale=2.5),
            actors={t.TrafficActor("car",speed=Distribution(mean=0.5, sigma=0.8)): 1},
        )
        for _ in range(3)
    for _ in range(3)
    ]
)
ego_mission = [t.Mission(t.Route(begin=("E8",0,1),end=("E6",0,'max')),start_time=5)]

scenario = t.Scenario(
    traffic={"all": traffic},
    ego_missions=ego_mission,
)

gen_scenario(scenario, output_dir=str(Path(__file__).parent))

# 1. exp distribution for for traffic, after 2 secs release one vehicle for lane0, 1 sec release on lane1...
# 2. uniformly select traffic density for each scenario
# 3. randomize the order of scenarios in env.