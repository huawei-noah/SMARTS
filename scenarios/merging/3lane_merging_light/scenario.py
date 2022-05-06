from pathlib import Path

import random
from smarts.sstudio import gen_scenario
from smarts.sstudio import types as t
from smarts.sstudio.types import Distribution
traffic = t.Traffic(
    flows=[
        t.Flow(
            route=t.Route(
                begin=("E5", random.randint(0,2), "random"),
                end=("E6", random.randint(0,2), "max"),
            ),
            rate=3,
            actors={t.TrafficActor("car",speed=Distribution(mean=1, sigma=1)): 1},
        )
        for _ in range(3)
    for _ in range(5)
    ]
)
ego_mission = [t.Mission(t.Route(begin=("E8",0,1),end=("E6",0,'max')))]

scenario = t.Scenario(
    traffic={"all": traffic},
    ego_missions=ego_mission,
)

gen_scenario(scenario, output_dir=str(Path(__file__).parent))
