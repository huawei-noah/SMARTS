import os
from pathlib import Path

from smarts.sstudio.genscenario import gen_scenario
from smarts.sstudio.types import (
    EndlessMission,
    Flow,
    JunctionEdgeIDResolver,
    Mission,
    RandomRoute,
    Route,
    Scenario,
    Traffic,
    TrafficActor,
    Via,
    CutIn,
)

ego_missions = [
    Mission(
        route=Route(begin=('edge-west-WE', 0, 50), end=('edge-east-WE', 1, 100)),
        task=CutIn(),
    ),
]

scenario = Scenario(
    traffic={
        "basic": Traffic(
            flows=[
                Flow(
                    route=Route(begin=('edge-west-WE', 1, 40), end=('edge-east-WE', 1, 100)),
                    rate=400,
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
