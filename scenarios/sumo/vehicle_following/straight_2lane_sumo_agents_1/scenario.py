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
    depart_speed=0,
    speed=Distribution(mean=1, sigma=0.2),
)

leader = TrafficActor(name="Leader-007", depart_speed=0)

# Social path = (start_lane, end_lane)
social_paths = [
    (0, 0),
    (0, 1),
    (1, 0),
    (1, 1),
]
min_flows = 2
max_flows = 3
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
        flows=[],
        trips=[
            Trip(
                vehicle_name="Leader-007",
                route=Route(
                    begin=("E0", 1, 15),
                    end=("E1.243", leader_path[1], "max"),
                ),
                actor=leader,
                vehicle_type="truck",
            ),
        ],
    )


default_speed = 13
route_length = 336
duration = (route_length / default_speed) * 2

ego_missions = [
    EndlessMission(
        begin=("E0", 1, 5),
        entry_tactic=TrapEntryTactic(
            start_time=1, wait_to_hijack_limit_s=0, default_entry_speed=0
        ),
    )  # Delayed start, to ensure road has prior traffic.
]
leader_id = "Leader-007"
gen_scenario(
    scenario=Scenario(
        traffic=traffic,
        ego_missions=ego_missions,
        scenario_metadata=ScenarioMetadata(
            actor_of_interest_re_filter=leader_id,
            actor_of_interest_color=Colors.Blue,
            scenario_difficulty=0.3,
            scenario_duration=duration,
        ),
    ),
    output_dir=Path(__file__).parent,
)
