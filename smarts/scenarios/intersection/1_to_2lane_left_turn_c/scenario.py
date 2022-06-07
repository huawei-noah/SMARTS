import random
import itertools
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
    (0, 1),
    (1, 0),
]
start_edges=[("E0",1),("E4",2),("-E1",2),("-E3",1)]
end_edges=[("-E0",1),("-E4",2),("E1",2),("E3",1)]
route_options=list(itertools.product(start_edges, end_edges))
num_routes = 5
num_configurations = 4
# Traffic combinations = 3C1 + 3C2 + 3C3 = 3 + 3 + 1 = 7
traffic = {}
for name in range(num_configurations):
    routes = []
    while len(routes) < num_routes:
        choice = random.choice(route_options)
        start_edge, end_edge = choice
        if start_edge[0] in end_edge[0] or end_edge[0] in start_edge[0]:
            continue
        routes.append(choice)
    traffic[str(name)] = Traffic(
        flows=[
            Flow(
                route=Route(
                    begin=(start_edge, random.randint(0, se_lanes - 1), 0),
                    end=(end_edge, random.randint(0, ee_lanes - 1), "max"),
                ),
                # Random flow rate, between x and y vehicles per minute.
                rate=random.uniform(10, 30),
                # Random flow start time, between x and y seconds.
                begin=random.uniform(0, 7),
                # For an episode with maximum_episode_steps=3000 and step
                # time=0.1s, maximum episode time=300s. Hence, traffic set to
                # end at 900s, which is greater than maximum episode time of
                # 300s.
                end=60 * 15,
                actors={normal: 1},
            )
            for (start_edge, se_lanes), (end_edge, ee_lanes) in routes
        ]
    )

route = Route(begin=("E0", 0, 1), end=("E1", 0, "max"))
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
