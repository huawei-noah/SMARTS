from pathlib import Path

from smarts.sstudio.genscenario import gen_scenario
from smarts.sstudio.types import (
    Flow,
    Mission,
    Route,
    Scenario,
    Traffic,
    TrafficActor,
)

scnr_path = str(Path(__file__).parent)

intersection_car = TrafficActor(
    name="car",
)

vertical_routes = [
    ("north-NS", "south-NS", 0),
    ("south-SN", "north-SN", 0),
]

horizontal_routes = [
    ("west-WE", "east-WE", "random"),
    ("east-EW", "west-EW", 0),
]

turn_left_routes = [
    ("south-SN", "west-EW", 0),
    ("west-WE", "north-SN", "random"),
    ("north-NS", "east-WE", 0),
    ("east-EW", "south-NS", 0),
]

turn_right_routes = [
    ("south-SN", "east-WE", 0),
    ("west-WE", "south-NS", "random"),
    ("north-NS", "west-EW", 0),
    ("east-EW", "north-SN", 0),
]

traffic = {}
for name, routes in {
    # "vertical": vertical_routes,
    "horizontal": horizontal_routes,
    # "turn_left": turn_left_routes,
    # "turn_right": turn_right_routes,
    # "turns": turn_left_routes + turn_right_routes,
    # "all": vertical_routes + horizontal_routes + turn_left_routes + turn_right_routes,
}.items():
    traffic[name] = Traffic(
        flows=[
            Flow(
                route=Route(
                    begin=(f"edge-{r[0]}", 0, r[2]),
                    end=(f"edge-{r[1]}", 0, "max"),
                ),
                rate=60 * 3,
                actors={intersection_car:1},
            )
            for i, r in enumerate(routes)
        ]
    )

ego_missions = [
    Mission(
        route=Route(begin=("edge-west-WE", 0, 40), end=("edge-north-SN", 0, 30)),
    ),
]

gen_scenario(
    scenario=Scenario(
        traffic=traffic,
        ego_missions=ego_missions,
    ),
    output_dir=scnr_path,
)
