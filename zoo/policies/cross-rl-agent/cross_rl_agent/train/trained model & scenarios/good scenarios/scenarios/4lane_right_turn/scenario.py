from pathlib import Path

from smarts.sstudio.genscenario import gen_scenario
from smarts.sstudio.types import (
    Flow,
    Mission,
    RandomRoute,
    Route,
    Scenario,
    Traffic,
    TrafficActor,
    Distribution,
    LaneChangingModel,
)

social_vehicle_num = 100

ego_missions = [
    Mission(route=Route(begin=("edge-south-SN", 0, 10), end=("edge-east-WE", 0, 8)),),
]

right_traffic_actor = TrafficActor(
    name="car",
    speed=Distribution(sigma=0.2, mean=1),
    lane_changing_model=LaneChangingModel(impatience=0),
)
scenario = Scenario(
    traffic={
        "basic": Traffic(
            flows=[
                Flow(route=RandomRoute(), rate=1, actors={right_traffic_actor: 1.0},)
                for i in range(social_vehicle_num)
            ]
        )
    },
    ego_missions=ego_missions,
)

gen_scenario(scenario=scenario, output_dir=Path(__file__).parent, ovewrite=True)
