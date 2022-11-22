from pathlib import Path

from smarts.sstudio.genscenario import gen_scenario
from smarts.sstudio.types import (
    Distribution,
    Flow,
    JunctionModel,
    LaneChangingModel,
    MapSpec,
    Route,
    Scenario,
    Traffic,
    TrafficActor,
)

impatient_car = TrafficActor(
    name="car",
    speed=Distribution(sigma=0.2, mean=1.0),
    lane_changing_model=LaneChangingModel(impatience=1, cooperative=0.25),
    junction_model=JunctionModel(
        drive_after_red_time=1.5, drive_after_yellow_time=1.0, impatience=1.0
    ),
)

patient_car = TrafficActor(
    name="car",
    speed=Distribution(sigma=0.2, mean=0.8),
    lane_changing_model=LaneChangingModel(impatience=0, cooperative=0.5),
    junction_model=JunctionModel(drive_after_yellow_time=1.0, impatience=0.5),
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

traffic = {}

for name, routes in {
    "vertical": vertical_routes,
    "horizontal": horizontal_routes,
    "unprotected_left": turn_left_routes,
    "turns": turn_left_routes + turn_right_routes,
    "all": vertical_routes + horizontal_routes + turn_left_routes + turn_right_routes,
}.items():
    traffic[name] = Traffic(
        flows=[
            Flow(
                route=Route(
                    begin=(f"edge-{r[0]}", 0, "random"),
                    end=(f"edge-{r[1]}", 0, "random"),
                ),
                repeat_route=True,
                rate=1,
                actors={impatient_car: 0.5, patient_car: 0.5},
            )
            for r in routes
        ]
    )

gen_scenario(
    scenario=Scenario(
        traffic=traffic,
        map_spec=MapSpec(
            source=Path(__file__).parent,
            shift_to_origin=True,
        ),
    ),
    output_dir=Path(__file__).parent,
)
