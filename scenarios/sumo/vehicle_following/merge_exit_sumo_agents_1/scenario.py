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
    MapSpec,
    Mission,
)

normal = TrafficActor(
    name="car",
    speed=Distribution(sigma=0, mean=1.0),
)

leader = TrafficActor(
    name="Leader-007",
    depart_speed=0,
    speed=Distribution(sigma=0.2, mean=0.8),
)

# Social path = (start_lane, end_lane)
social_paths = [
    (0, 0),
    (0, 1),
    (0, 2),
    (1, 0),
    (1, 1),
    (1, 2),
    (2, 0),
    (2, 1),
    (2, 2),
]
min_flows = 2
max_flows = 4
social_comb = [
    com
    for elems in range(min_flows, max_flows + 1)
    for com in combinations(social_paths, elems)
]

# Leader path = (start_lane, end_lane)
leader_paths = [0, 1, 2]

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
                    begin=("E0", leader_path, 25),
                    end=("E4", 0, "max"),
                ),
                actor=leader,
                vehicle_type="truck",
            ),
        ],
    )
ego_missions = []
lane_idx = [0, 1, 2]
for i in lane_idx:
    ego_missions.append(
        EndlessMission(
            begin=("E0", i, 5),
            start_time=1,
            entry_tactic=TrapEntryTactic(
                wait_to_hijack_limit_s=0, default_entry_speed=0
            ),
        )
    )

gen_scenario(
    scenario=Scenario(
        traffic=traffic,
        ego_missions=ego_missions,
        map_spec=MapSpec(
            source=Path(__file__).resolve().parents[0], lanepoint_spacing=1.0
        ),
        scenario_metadata=ScenarioMetadata("Leader-007", Colors.Blue),
    ),
    output_dir=Path(__file__).parent,
)
