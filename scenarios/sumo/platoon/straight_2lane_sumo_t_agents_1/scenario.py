import random
from itertools import combinations, product
from pathlib import Path

from smarts.core.colors import Colors
from smarts.sstudio import gen_scenario
from smarts.sstudio.types import (
    Distribution,
    EndlessMission,
    Flow,
    Route,
    Scenario,
    ScenarioMetadata,
    Traffic,
    TrafficActor,
    TrapEntryTactic,
    Trip,
)

normal = TrafficActor(
    name="car",
    speed=Distribution(sigma=0.5, mean=1.0),
)

leader = TrafficActor(
    name="Leader-007",
    depart_speed=0,
)

# Social path = (start_lane, end_lane)
social_paths = [
    (0, 0),
    (0, 1),
    (1, 0),
    (1, 1),
]
min_flows = 2
max_flows = 4
social_comb = [
    com
    for elems in range(min_flows, max_flows + 1)
    for com in combinations(social_paths, elems)
] * 10

# Leader path = (start_lane, end_lane)
leader_paths = [(0, 0), (0, 1)]

# Overall routes
route_comb = product(social_comb, leader_paths)

traffic = {}
for name, (social_path, leader_path) in enumerate(route_comb):
    traffic[str(name)] = Traffic(
        engine="SUMO",
        flows=[
            Flow(
                route=Route(
                    begin=("E0", r[0], 0),
                    end=("E0", r[1], "max"),
                ),
                # Random flow rate, between x and y vehicles per minute.
                rate=60 * random.uniform(4, 6),
                # Random flow start time, between x and y seconds.
                begin=random.uniform(0, 5),
                # For an episode with maximum_episode_steps=3000 and step
                # time=0.1s, maximum episode time=300s. Hence, traffic set to
                # end at 900s, which is greater than maximum episode time of
                # 300s.
                end=60 * 15,
                actors={normal: 1},
                randomly_spaced=True,
            )
            for r in social_path
        ],
        trips=[
            Trip(
                vehicle_name="Leader-007",
                route=Route(
                    begin=("E0", 1, 20),
                    end=("E0", leader_path[1], "max"),
                ),
                depart=19,
                actor=leader,
                vehicle_type="truck",
            ),
        ],
    )


ego_missions = [
    EndlessMission(
        begin=("E0", 1, 10),
        start_time=20,
        entry_tactic=TrapEntryTactic(wait_to_hijack_limit_s=0, default_entry_speed=0),
    )  # Delayed start, to ensure road has prior traffic.
]

gen_scenario(
    scenario=Scenario(
        traffic=traffic,
        ego_missions=ego_missions,
        scenario_metadata=ScenarioMetadata("Leader-007", Colors.Blue),
    ),
    output_dir=Path(__file__).parent,
)
