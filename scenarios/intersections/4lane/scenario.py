import os
from pathlib import Path

from smarts.sstudio.genscenario import gen_scenario
from smarts.sstudio.types import (
    EndlessMission,
    Flow,
    Mission,
    RandomRoute,
    Route,
    Scenario,
    Traffic,
    TrafficActor,
    Via,
)

ego_missions = [
    Mission(
        route=Route(begin=("edge-south-SN", 1, 10), end=("edge-west-EW", 1, "max")),
    ),
    Mission(
        route=Route(begin=("edge-west-WE", 1, 10), end=("edge-east-WE", 1, "max")),
    ),
]

scenario = Scenario(
    traffic={
        "basic": Traffic(
            flows=[
                Flow(
                    route=RandomRoute(),
                    rate=3600,
                    actors={TrafficActor(name="car"): 1.0},
                )
            ]
        )
    },
    ego_missions=ego_missions,
)

gen_scenario(
    scenario=scenario, output_dir=Path(__file__).parent,
)
