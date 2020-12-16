from pathlib import Path

from smarts.sstudio import gen_missions, gen_traffic
from smarts.sstudio.types import (
    Route,
    Mission,
    CutIn,
)
from smarts.sstudio import types as t


scenario = str(Path(__file__).parent)

traffic = t.Traffic(
    flows=[
        t.Flow(
            route=Route(begin=("gneE20", 0, 55), end=("gneE20", 0, "max")),
            rate=1,
            actors={t.TrafficActor("car", max_speed=50/3.6): 1},
        )
    ],
)

gen_traffic(
    scenario=scenario,
    traffic=traffic,
    name="basic"
)


gen_missions(
    scenario=scenario,
    missions=[
        Mission(Route(begin=("gneE20", 1, 85), end=("gneE20", 0, "max")), task=CutIn()),
    ],
)
