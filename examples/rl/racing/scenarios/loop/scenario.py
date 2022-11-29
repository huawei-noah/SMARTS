from pathlib import Path

from smarts.sstudio import gen_scenario
from smarts.sstudio import types as t

traffic = t.Traffic(
    flows=[
        t.Flow(
            route=t.RandomRoute(),
            rate=1,
            end=10,  # `rate=1` adds 1 additional vehicle per hour. So set `end` < 1*60*60 secs to avoid addition of more vehicles after the initial flow. This prevents traffic congestion.
            actors={t.TrafficActor(name="car", vehicle_type=vehicle_type): 1},
        )
        for vehicle_type in [
            "passenger",
            "bus",
            "coach",
            "truck",
            "trailer",
            "passenger",
            "bus",
            "coach",
            "truck",
            "trailer",
        ]
        * 2
    ]
)

gen_scenario(
    t.Scenario(
        traffic={"basic": traffic},
    ),
    output_dir=Path(__file__).parent,
)
