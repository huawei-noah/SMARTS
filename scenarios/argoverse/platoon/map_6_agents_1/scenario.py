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

PATH = "/home/kyber/workspace/argoverse_data/train"
scenario_id = "2608ed4d-57ac-4398-b8b8-df3a2febbe80"
scenario_path = Path(PATH) / scenario_id

start_road = "road-216154773-216154779-216154776"
lane_idx = (0,)
end_road = ("road-216075790", "road-216077525", "road-216075807-216076742-216076861")

route_comb = product(lane_idx, end_road)
leader_mission = []
for route in route_comb:
    leader_mission.append(
        Mission(
            Route(begin=(start_road, 0, 10), end=(route[1], 0, "max")),
        )
    )

ego_missions = [
    EndlessMission(
        begin=(start_road, 0, 5),
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
