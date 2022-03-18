import os

from smarts.sstudio import gen_traffic
from smarts.sstudio.types import (
    Distribution,
    Flow,
    SumoVTypeOverride,
    Route,
    Traffic,
    TrafficActor,
)

scenario = os.path.dirname(os.path.realpath(__file__))

impatient_car = TrafficActor(
    name="car",
    speed=Distribution(sigma=0.2, mean=1.0),
    model_overrides=(
        SumoVTypeOverride(
            lcImpatience=1,
            lcCooperative=0.25,
            jmDriveAfterRedTime=1.5,
            jmDriveAfterYellowTime=1.0,
            impatience=1.0,
        ),
    ),
)

patient_car = TrafficActor(
    name="car",
    speed=Distribution(sigma=0.2, mean=0.8),
    model_overrides=(
        SumoVTypeOverride(
            lcImpatience=0,
            lcCooperative=0.5,
            jmDriveAfterYellowTime=1.0,
            impatience=0.5,
        ),
    ),
)

vertical_routes = [
    ("north-NS", "south-NS"),
    ("south-SN", "north-SN"),
]

horizontal_routes = [
    ("west-WE", "east-WE"),
    ("east-EW", "west-EW"),
]

turn_left_routes = [
    ("south-SN", "west-EW"),
    ("west-WE", "north-SN"),
    ("north-NS", "east-WE"),
    ("east-EW", "south-NS"),
]

turn_right_routes = [
    ("south-SN", "east-WE"),
    ("west-WE", "south-NS"),
    ("north-NS", "west-EW"),
    ("east-EW", "north-SN"),
]

for name, routes in {
    "vertical": vertical_routes,
    "horizontal": horizontal_routes,
    "unprotected_left": turn_left_routes,
    "turns": turn_left_routes + turn_right_routes,
    "all": vertical_routes + horizontal_routes + turn_left_routes + turn_right_routes,
}.items():
    traffic = Traffic(
        flows=[
            Flow(
                route=Route(
                    begin=(f"edge-{r[0]}", 0, "random"),
                    end=(f"edge-{r[1]}", 0, "random"),
                ),
                rate=1,
                actors={impatient_car: 0.5, patient_car: 0.5},
            )
            for r in routes
        ]
    )

    gen_traffic(scenario, traffic, name=name)
