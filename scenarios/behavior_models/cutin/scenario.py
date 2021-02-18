import os
from pathlib import Path

from smarts.sstudio.genscenario import gen_scenario
from smarts.sstudio.types import (
    CutIn,
    Flow,
    Mission,
    Route,
    Scenario,
    Traffic,
    TrafficActor,
)

ego_missions = [
    Mission(
        route=Route(begin=("edge-west-WE", 0, 10), end=("edge-west-WE", 0, "max")),
        task=CutIn(),
    ),
]

scenario = Scenario(
    traffic={
        "basic": Traffic(
            flows=[
                Flow(
                    route=Route(
                        begin=("edge-west-WE", 1, 10), end=("edge-west-WE", 1, "max")
                    ),
                    rate=400,
                    actors={TrafficActor(name="car"): 1.0},
                )
            ]
        )
    },
    ego_missions=ego_missions,
)

gen_scenario(
    scenario=scenario,
    output_dir=Path(__file__).parent,
)
