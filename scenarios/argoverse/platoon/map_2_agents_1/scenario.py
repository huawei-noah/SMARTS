from pathlib import Path
from itertools import product
from pathlib import Path

from smarts.sstudio.types import (
    Route,
    Mission,
    SocialAgentActor,
    EndlessMission,
    TrapEntryTactic,
)

from smarts.sstudio import gen_scenario
from smarts.sstudio import types as t

# scenario_path is a directory with the following structure:
# /path/to/dataset/{scenario_id}
# ├── log_map_archive_{scenario_id}.json
# └── scenario_{scenario_id}.parquet

PATH = "/home/kyber/workspace/argoverse_data/train"
scenario_id = "0a0a71c5-2e02-432c-ade6-dff22bc659de"  # e.g. "0000b6ab-e100-4f6b-aee8-b520b57c0530"
scenario_path = Path(PATH) / scenario_id  # e.g. Path("/home/user/argoverse/train/") / scenario_id

first_via = (0, 1, 2)
second_via = (0, 1, 2)
third_via = (0, 1, 2)
lane_idx = (0,1)
start_road = ("road-393322800-393322556","road-393323181","road-393323224","road-393323350")
end_road = ("road-393323082-393322772","road-393323379","road-393323179-393323328")

route_comb = product(start_road, lane_idx,first_via, second_via, third_via,end_road)
leader_mission = []
for route in route_comb:
    leader_mission.append(
        Mission(Route(
            begin=(route[0],0,10),end=(route[5],0,"max")),
        )
    )

leader_actor = [
    SocialAgentActor(
        name="Leader-007",
        agent_locator="zoo.policies:chase-via-points-agent-v0",
        initial_speed=1,
    )
]

ego_missions = [
    EndlessMission(
        begin=("E_start", 1, 5), 
        start_time=1,
        entry_tactic=TrapEntryTactic(
            wait_to_hijack_limit_s=1,
            default_entry_speed=1,
        ),
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
        social_agent_missions={"leader":(leader_actor, leader_mission)},
        map_spec=t.MapSpec(source=f"{scenario_path}", lanepoint_spacing=1.0),
        # traffic_histories=traffic_histories,
    ),
    output_dir=Path(__file__).parent,
)
