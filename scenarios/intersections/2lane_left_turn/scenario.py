import random
from itertools import combinations
from pathlib import Path

from smarts.sstudio.genscenario import gen_scenario
from smarts.sstudio.types import (
    Flow,
    MapZone,
    Mission,
    Route,
    Scenario,
    Traffic,
    TrafficActor,
    TrapEntryTactic,
)

intersection_car = TrafficActor(
    name="car",
)

vertical_routes = [
    ("north-NS", "south-NS"),
    ("south-SN", "north-SN"),
]

horizontal_routes = [
    ("west-WE", "east-WE"),
    ("east-EW", "west-EW"),
]

turn_left_routes = [
    ("south-SN", "west-EW"),
    ("west-WE", "north-SN"),
    ("north-NS", "east-WE"),
    ("east-EW", "south-NS"),
]

turn_right_routes = [
    ("south-SN", "east-WE"),
    ("west-WE", "south-NS"),
    ("north-NS", "west-EW"),
    ("east-EW", "north-SN"),
]

# Total route combinations = 12C1 + 12C2 + 12C3 + 12C4 = 793
all_routes = vertical_routes + horizontal_routes + turn_left_routes + turn_right_routes
route_comb = [com for elem in range(1, 5) for com in combinations(all_routes, elem)]
route_comb = random.shuffle(route_comb)
traffic = {}
for name, routes in enumerate(route_comb):
    traffic[name] = Traffic(
        flows=[
            Flow(
                route=Route(
                    begin=(f"edge-{r[0]}", 0, 0),
                    end=(f"edge-{r[1]}", 0, "max"),
                ),
                # Random flow rate, between 1 vehicle per minute and 6 vehicles
                # per minute.
                rate=1,  # 60 * random.uniform(1, 6),
                begin=random.uniform(0, 10),
                end=60 * 15,
                # For an episode with maximum_episode_steps=3000 and step
                # time=0.1s, maximum episode time=300s. Hence, traffic set to
                # end at 900s, which is greater than maximum episode time of
                # 300s.
                actors={intersection_car: 1},
            )
            for r in routes
        ]
    )

route = Route(begin=("edge-west-WE", 0, 60), end=("edge-north-SN", 0, 40))
ego_missions = [
    Mission(
        route=route,
        start_time=15,  # Delayed start, to ensure road has prior traffic.
        entry_tactic=TrapEntryTactic(
            wait_to_hijack_limit_s=1,
            zone=MapZone(
                start=(
                    route.begin[0],
                    route.begin[1],
                    route.begin[2] - 5,
                ),
                length=10,
                n_lanes=1,
            ),
            default_entry_speed=5,
        ),
    ),
]

scnr_path = Path(__file__).parent
gen_scenario(
    scenario=Scenario(
        traffic=traffic,
        ego_missions=ego_missions,
    ),
    output_dir=scnr_path,
)
