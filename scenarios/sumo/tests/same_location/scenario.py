from pathlib import Path
from typing import Any, Tuple

import smarts.sstudio.types as types
from smarts.sstudio import gen_missions, gen_traffic

scenario = str(Path(__file__).parent)

patient_car = types.TrafficActor(
    name="car",
)

shared_route = types.Route(
    begin=("edge-east", 0, 20),
    end=("edge-west", 0, 0),
)

traffic = types.Traffic(
    flows=[
        types.Flow(
            route=shared_route,
            rate=1,
            actors={patient_car: 1},
        )
    ]
)

gen_missions(
    scenario,
    missions=[
        types.Mission(shared_route),
    ],
)
gen_traffic(scenario, traffic, "traffic")
