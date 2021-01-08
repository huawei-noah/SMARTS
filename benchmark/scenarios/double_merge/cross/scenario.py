from pathlib import Path
from smarts.sstudio import gen_scenario
import smarts.sstudio.types as t

missions = [
    t.Mission(t.Route(begin=("gneE17", 0, 10), end=("gneE5", 0, 100))),
    t.Mission(t.Route(begin=("gneE22", 0, 10), end=("gneE5", 1, 100))),
    t.Mission(t.Route(begin=("gneE17", 0, 25), end=("gneE5", 0, 100))),
    t.Mission(t.Route(begin=("gneE22", 0, 25), end=("gneE5", 1, 100))),
]

impatient_car = t.TrafficActor(
    name="car",
    speed=t.Distribution(sigma=0.2, mean=1.0),
    lane_changing_model=t.LaneChangingModel(impatience=1, cooperative=0.25),
    junction_model=t.JunctionModel(
        drive_after_red_time=1.5, drive_after_yellow_time=1.0, impatience=1.0
    ),
)

patient_car = t.TrafficActor(
    name="car",
    speed=t.Distribution(sigma=0.2, mean=0.8),
    lane_changing_model=t.LaneChangingModel(impatience=0, cooperative=0.5),
    junction_model=t.JunctionModel(drive_after_yellow_time=1.0, impatience=0.5),
)


traffic = {
    "1": t.Traffic(
        flows=[
            t.Flow(
                route=t.Route(begin=(f"gneE17", 0, 2), end=(f"gneE5", 1, 120),),
                rate=1,
                actors={impatient_car: 0.5, patient_car: 0.5},
            )
        ]
    ),
    "2": t.Traffic(
        flows=[
            t.Flow(
                route=t.Route(begin=(f"gneE22", 0, 2), end=(f"gneE5", 0, 120),),
                rate=1,
                actors={impatient_car: 0.5, patient_car: 0.5},
            )
        ]
    ),
}

gen_scenario(
    t.Scenario(ego_missions=missions, traffic=traffic),
    output_dir=Path(__file__).parent,
)
