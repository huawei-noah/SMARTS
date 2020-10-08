import os

from smarts.sstudio import gen_traffic, gen_missions
from smarts.sstudio.types import (
    Traffic,
    Flow,
    Route,
    RandomRoute,
    TrafficActor,
    Mission,
)

scenario = os.path.dirname(os.path.realpath(__file__))

gen_missions(
    scenario,
    [
        Mission(Route(begin=("edge-west-WE", 0, 10), end=("edge-south-NS", 0, 40))),
        Mission(Route(begin=("edge-west-WE", 1, 10), end=("edge-south-NS", 0, 40))),
    ],
)

gen_traffic(
    scenario,
    Traffic(
        flows=[
            Flow(
                route=RandomRoute(), rate=3600, actors={TrafficActor(name="car"): 1.0},
            )
        ]
    ),
    name="random",
)
