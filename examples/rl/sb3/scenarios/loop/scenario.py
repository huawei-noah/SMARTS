from pathlib import Path

from smarts.sstudio import gen_scenario
from smarts.sstudio import types as t

traffic = t.Traffic(
    flows=[
        t.Flow(
            route=t.RandomRoute(),
            rate=60 * 60,
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
    ]
)

gen_scenario(
    t.Scenario(
        traffic={"basic": traffic},
    ),
    output_dir=Path(__file__).parent,
)
