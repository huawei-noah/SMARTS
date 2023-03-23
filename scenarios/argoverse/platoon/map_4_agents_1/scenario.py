from itertools import product
from pathlib import Path

from smarts.sstudio import gen_scenario
from smarts.sstudio import types as t
from smarts.sstudio.types import (
    EndlessMission,
    Mission,
    Route,
    SocialAgentActor,
    TrapEntryTactic,
)

PATH = "argoverse/data"
scenario_id = "c627814f-8880-4142-91c9-96b104c1bece"
scenario_path = Path(__file__).resolve().parents[5] / PATH / scenario_id

end_road = (
    ("road-394975904", 0),
    ("road-394976408-394976268", 0),
    ("road-394969208-394969285", 1),
    ("road-394976534", 0),
)

route_comb = end_road
leader_mission = []
for route in route_comb:
    leader_mission.append(
        Mission(
            Route(begin=("road-394969256", 0, 10), end=(route[0], route[1], "max")),
        )
    )

ego_missions = [
    EndlessMission(
        begin=("road-394969256", 0, 5),
        entry_tactic=TrapEntryTactic(
            wait_to_hijack_limit_s=0,
            default_entry_speed=1,
        ),
    )
]

leader_actor = [
    SocialAgentActor(
        name="Leader-007",
        agent_locator="zoo.policies:chase-via-points-agent-v0",
        initial_speed=1,
    )
]

traffic_histories = [
    t.TrafficHistoryDataset(
        name=f"argoverse_{scenario_id}",
        source_type="Argoverse",
        input_path=scenario_path,
    )
]

gen_scenario(
    t.Scenario(
        social_agent_missions={"leader": (leader_actor, leader_mission)},
        ego_missions=ego_missions,
        map_spec=t.MapSpec(source=f"{scenario_path}", lanepoint_spacing=1.0),
        # traffic_histories=traffic_histories,
    ),
    output_dir=Path(__file__).parent,
)
