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
    UTurn,
)

ego_missions = [
    Mission(
        route=Route(begin=('edge-west-WE', 0, 50), end=('edge-west-EW', 0, 100)),
        task=UTurn(initial_speed=10),
    ),
]

scenario = Scenario(
    traffic={
        "basic": Traffic(
            flows=[
                Flow(
                    route=Route(begin=('edge-west-EW', 0, 0), end=('edge-west-EW', 1, 'max')),
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
