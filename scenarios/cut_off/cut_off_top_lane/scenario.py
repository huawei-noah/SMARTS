from pathlib import Path

import random
import numpy as np
from smarts.sstudio import gen_scenario
from smarts.sstudio import types as t
from smarts.sstudio.types import Distribution
traffic = t.Traffic(
    flows=[
        t.Flow(
            route=t.Route(
                begin=("E1", random.randint(0,2), "random"),
                end=("E2", random.randint(0,2), "max"),
            ),
            rate=3,
            begin=np.random.exponential(scale=2.5),
            actors={t.TrafficActor("car",speed=Distribution(mean=0.5, sigma=0.8)): 1},
        )
        for _ in range(3)
    for _ in range(4)
    ]
)
ego_mission = [t.Mission(t.Route(begin=("E1",2,1),end=("E2",2,'max')))]

scenario = t.Scenario(
    traffic={"all": traffic},
    ego_missions=ego_mission,
)

gen_scenario(scenario, output_dir=str(Path(__file__).parent))
