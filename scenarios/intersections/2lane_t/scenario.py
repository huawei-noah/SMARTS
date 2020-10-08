import os

from smarts.sstudio import gen_traffic, gen_missions
from smarts.sstudio.types import (
    Traffic,
    Flow,
    Route,
    RandomRoute,
    TrafficActor,
    Distribution,
    LaneChangingModel,
    JunctionModel,
    Mission,
    TrapEntryTactic,
    MapZone,
)

scenario = os.path.dirname(os.path.realpath(__file__))

gen_missions(
    scenario,
    [
        Mission(
            Route(begin=("edge-west-WE", 0, 10), end=("edge-south-NS", 0, 40)),
            entry_tactic=TrapEntryTactic(
                wait_to_hijack_limit_s=3,
                zone=MapZone(start=("edge-west-WE", 0, 5), length=30, n_lanes=1,),
            ),
        ),
        Mission(
            Route(begin=("edge-west-WE", 0, 10), end=("edge-west-WE", 0, 25)),
            entry_tactic=TrapEntryTactic(
                wait_to_hijack_limit_s=3,
                zone=MapZone(start=("edge-west-WE", 0, 5), length=30, n_lanes=1,),
            ),
        ),
        Mission(
            Route(begin=("edge-south-SN", 0, 30), end=("edge-west-EW", 0, 50)),
            entry_tactic=TrapEntryTactic(
                wait_to_hijack_limit_s=3,
                zone=MapZone(start=("edge-south-SN", 0, 5), length=30, n_lanes=1,),
            ),
        ),
    ],
)

total_rate = 3600  # vehicles/hr

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

for name, routes in {
    "left_to_right": [("west-WE", "east-WE"),],
    "two_way": [("west-WE", "east-WE"), ("east-EW", "west-EW"),],
    "all": [
        ("west-WE", "east-WE"),
        ("east-EW", "west-EW"),
        ("west-WE", "south-NS"),
        ("east-EW", "south-NS"),
    ],
}.items():
    traffic = Traffic(
        flows=[
            Flow(
                route=Route(
                    begin=(f"edge-{r[0]}", 0, 20), end=(f"edge-{r[1]}", 0, -20)
                ),
                rate=total_rate,
                # Share the total rate amongst all flows
                actors={
                    impatient_car: (1.0 / len(routes)) * 0.5,
                    patient_car: (1.0 / len(routes)) * 0.5,
                },
            )
            for r in routes
        ]
    )

    gen_traffic(scenario, traffic, name=name)

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
