import random
from itertools import combinations
from pathlib import Path

from smarts.sstudio import gen_scenario
from smarts.sstudio.types import (
    Distribution,
    Flow,
    Mission,
    Route,
    Scenario,
    Traffic,
    TrafficActor,
)

normal = TrafficActor(
    name="car",
    speed=Distribution(sigma=0.8, mean=1),
)

# flow_name = (start_lane, end_lane,)
route_opt = [
    (0, 0),
    (1, 1),
    (2, 2),
]

# Traffic combinations = 3C1 + 3C2 + 3C3 = 3 + 3 + 1 = 7
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
                    begin=("gneE3", r[0], 0),
                    end=("gneE3", r[1], "max"),
                ),
                # Random flow rate, between x and y vehicles per minute.
                rate=10 * random.uniform(3, 5),
                # Random flow start time, between x and y seconds.
                # begin=random.uniform(0, 7),
                # For an episode with maximum_episode_steps=3000 and step
                # time=0.1s, maximum episode time=300s. Hence, traffic set to
                # end at 900s, which is greater than maximum episode time of
                # 300s.
                # end=60 * 15,
                actors={normal: 1},
            )
            for r in routes
        ]
    )

missions = [
    Mission(Route(begin=("gneE3", 0, 10), end=("gneE3", 0, "max")),start_time=19),
    Mission(Route(begin=("gneE3", 1, 10), end=("gneE3", 1, "max")),start_time=21),
    Mission(Route(begin=("gneE3", 2, 10), end=("gneE3", 2, "max")),start_time=15),
]

gen_scenario(
    scenario=Scenario(
        traffic=traffic,
        ego_missions=missions,
    ),
    output_dir=Path(__file__).parent,
)
