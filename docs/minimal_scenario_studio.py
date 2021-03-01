from pathlib import Path

from smarts.sstudio import types as t
from smarts.sstudio import gen_traffic, gen_bubbles, gen_missions

scenario = str(Path(__file__).parent)

# Definition of a traffic flow
traffic = t.Traffic(
    flows=[
        t.Flow(
            route=t.Route(
                begin=("west", lane_idx, 10),
                end=("east", lane_idx, -10),
            ),
            rate=50,
            actors={
                t.TrafficActor("car"): 1,
            },
        )
        for lane_idx in range(3)
    ]
)

# The call to generate traffic
gen_traffic(scenario, traffic, name="basic")

gen_missions(
    scenario,
    [
        t.Mission(t.Route(begin=("west", 0, 0), end=("east", 0, "max"))),
    ],
)
