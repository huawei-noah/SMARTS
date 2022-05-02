from pathlib import Path

import random
from smarts.sstudio import gen_scenario
from smarts.sstudio import types as t
from smarts.sstudio.types import Distribution
traffic = t.Traffic(
    flows=[
        t.Flow(
            route=t.Route(
                begin=("E1", lane_idx, "random"),
                end=("E2", random.randint(0,2), "max"),
            ),
            rate=3,
            actors={t.TrafficActor("car",speed=Distribution(mean=0.5, sigma=1)): 1},
        )
        for lane_idx in range(3)
    for _ in range(4)
    ]
)

scenario = t.Scenario(
    traffic={"all": traffic},
)

gen_scenario(scenario, output_dir=str(Path(__file__).parent))
