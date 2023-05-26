import random
from itertools import combinations
from pathlib import Path
from numpy import random

from smarts.sstudio import gen_scenario
from smarts.sstudio.types import (
    Distribution,
    Flow,
    Mission,
    Route,
    Scenario,
    SmartsLaneChangingModel,
    Traffic,
    TrafficActor,
)

normal = TrafficActor(
    name="car",
    sigma=1,
    speed=Distribution(sigma=0.3, mean=1.5),
    min_gap=Distribution(sigma=0, mean=1),
    lane_changing_model=SmartsLaneChangingModel(
        cutin_prob=1, assertive=10, dogmatic=True, slow_down_after=0.7, hold_period=5
    ),
)

# flow_name = (start_lane, end_lane,)
route_opt = [
    (0, 0),
    (1, 1),
]

# Traffic combinations = 3C2 + 3C3 = 3 + 1 = 4
# Repeated traffic combinations = 4 * 100 = 400
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
        engine="SMARTS",
        flows=[
            Flow(
                route=Route(
                    begin=("gneE3", r[0], 0),
                    end=("gneE3", r[1], "max"),
                ),
                # Random flow rate, between x and y vehicles per minute.
                rate=60 * random.uniform(5, 10),
                # Random flow start time, between x and y seconds.
                begin=random.uniform(0, 5),
                # For an episode with maximum_episode_steps=3000 and step
                # time=0.1s, maximum episode time=300s. Hence, traffic set to
                # end at 900s, which is greater than maximum episode time of
                # 300s.
                end=60,
                actors={normal: 1},
                randomly_spaced=True,
            )
            for r in routes
        ],
    )


route = Route(begin=("gneE3", 1, 5), end=("gneE3", 1, "max"))
ego_missions = [
    Mission(
        route=route,
        start_time=10,  # Delayed start, to ensure road has prior traffic.
    )
]

gen_scenario(
    scenario=Scenario(
        traffic=traffic,
        ego_missions=ego_missions,
    ),
    output_dir=Path(__file__).parent,
)
