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
    JunctionModel,
)

normal = TrafficActor(
    name="car",
    speed=Distribution(sigma=0, mean=1.0),
)

leader = TrafficActor(
    name="Leader-007",
    depart_speed=0,
    speed=Distribution(sigma=0.2, mean=0.8),
    junction_model=JunctionModel(impatience=1),
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
        flows=[
            Flow(
                route=Route(
                    begin=("E1", r[0], 0),
                    end=("E3", r[1], "max"),
                ),
                # Random flow rate, between x and y vehicles per minute.
                rate=60 * random.uniform(1, 3),
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
                    begin=("E0", leader_path, 25),
                    end=("E4", 0, "max"),
                ),
                depart=30,
                actor=leader,
                vehicle_type="truck",
            ),
        ],
    )

ego_missions = [
    EndlessMission(
        begin=("E0", 2, 5),
        start_time=31,
        entry_tactic=TrapEntryTactic(wait_to_hijack_limit_s=0, default_entry_speed=0),
    ),
    EndlessMission(
        begin=("E0", 1, 10),
        start_time=31,
        entry_tactic=TrapEntryTactic(wait_to_hijack_limit_s=0, default_entry_speed=0),
    ),
]

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
