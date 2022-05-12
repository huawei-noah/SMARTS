import random
from itertools import combinations
from pathlib import Path

from smarts.sstudio import gen_scenario
from smarts.sstudio.types import (
    Distribution,
    Flow,
    JunctionModel,
    LaneChangingModel,
    Mission,
    Route,
    Scenario,
    Traffic,
    TrafficActor,
    UniformDistribution,
)

# See SUMO doc
# Lane changing model
# https://sumo.dlr.de/docs/Definition_of_Vehicles%2C_Vehicle_Types%2C_and_Routes.html#lane-changing_models
# Junction model
# https://sumo.dlr.de/docs/Definition_of_Vehicles%2C_Vehicle_Types%2C_and_Routes.html#junction_model_parameters

normal = TrafficActor(
    name="car",
    speed=Distribution(sigma=0.8, mean=0.8),
)
# cooperative = TrafficActor(
#     name="cooperative",
#     speed=Distribution(sigma=0.3, mean=1.0),
#     lane_changing_model=LaneChangingModel(
#         pushy=0.1,
#         impatience=0.1,
#         cooperative=0.9,
#         speed_Gain=0.8,
#     ),
#     junction_model=JunctionModel(
#         impatience=0.1,
#     ),
# )
# aggressive = TrafficActor(
#     name="aggressive",
#     speed=Distribution(sigma=0.3, mean=1.0),
#     lane_changing_model=LaneChangingModel(
#         pushy=0.8,
#         impatience=1,
#         cooperative=0.1,
#         speed_Gain=2.0,
#     ),
#     junction_model=JunctionModel(
#         impatience=0.6,
#     ),
# )

# flow_name = (start_lane, end_lane,)
route_opt = [
    (0, 0),
    (1, 1),
    (2, 2),
    # (0,1),
    # (0,2),
    # (1,0),
    # (1,2),
    # (2,0),
    # (2,1)
]

# Route combinations = 3C1 + 3C2 + 3C3 = 3 + 3 + 1 = 7
min_flows = 1
max_flows = 3
route_comb = [
    com
    for elems in range(min_flows, max_flows + 1)
    for com in combinations(route_opt, elems)
]

traffic = {}
for name, routes in enumerate(route_comb):
    traffic[str(name)] = Traffic(
        flows=[
            Flow(
                route=Route(
                    begin=("E5", r[0], 0),
                    end=("E6", r[1], "max"),
                ),
                # Random flow rate, between x and y vehicles per minute.
                rate=60 * random.uniform(8, 15),
                # Random flow start time, between x and y seconds.
                begin=random.uniform(0, 7),
                # For an episode with maximum_episode_steps=3000 and step
                # time=0.1s, maximum episode time=300s. Hence, traffic set to
                # end at 900s, which is greater than maximum episode time of
                # 300s.
                end=60 * 15,
                actors={normal: 1},
            )
            for r in routes
        ]
    )

route = Route(begin=("E8", 0, 1), end=("E6", 0, "max"))
ego_missions = [
    Mission(
        route=route,
        start_time=19,  # Delayed start, to ensure road has prior traffic.
    )
]

gen_scenario(
    scenario=Scenario(
        traffic=traffic,
        ego_missions=ego_missions,
    ),
    output_dir=Path(__file__).parent,
)
