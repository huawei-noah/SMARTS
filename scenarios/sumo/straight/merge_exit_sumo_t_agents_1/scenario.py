import random
from itertools import combinations, product
from pathlib import Path

from smarts.core.colors import Colors
from smarts.sstudio import gen_scenario
from smarts.sstudio.sstypes import (
    Distribution,
    EndlessMission,
    Flow,
    MapSpec,
    Mission,
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
    speed=Distribution(sigma=0, mean=1.0),
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
    )
default_speed = 13
route_length = 600
duration = (route_length / default_speed) * 2
ego_missions = [
    Mission(
        route=Route(begin=("E0", 2, 5), end=("E4", 0, "max")),
        start_time=30,
        entry_tactic=TrapEntryTactic(
            start_time=30, wait_to_hijack_limit_s=0, default_entry_speed=0
        ),
    )  # Delayed start, to ensure road has prior traffic.
]

gen_scenario(
    scenario=Scenario(
        traffic=traffic,
        ego_missions=ego_missions,
        map_spec=MapSpec(
            source=Path(__file__).resolve().parents[0], lanepoint_spacing=1.0
        ),
        scenario_metadata=ScenarioMetadata(
            scenario_difficulty=0.9,
            scenario_duration=duration,
        ),
    ),
    output_dir=Path(__file__).parent,
)
