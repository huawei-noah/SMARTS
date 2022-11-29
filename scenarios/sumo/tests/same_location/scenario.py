from pathlib import Path

import smarts.sstudio.types as types
from smarts.sstudio.genscenario import _gen_agent_missions, _gen_traffic

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
            repeat_route=True,
            rate=1,
            actors={patient_car: 1},
        )
    ]
)

_gen_agent_missions(
    scenario,
    missions=[
        types.Mission(shared_route),
    ],
)
_gen_traffic(scenario, traffic, "traffic")
