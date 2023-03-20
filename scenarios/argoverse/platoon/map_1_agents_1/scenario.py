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
scenario_id = "00a0a3e0-1508-45f2-9cf5-e427e1446a33"  # e.g. "0000b6ab-e100-4f6b-aee8-b520b57c0530"
scenario_path = Path(PATH) / scenario_id  # e.g. Path("/home/user/argoverse/train/") / scenario_id


start_road_1 = ("road-265543685-265543751","road-265545149-265545151")
lane_idx_1 = (0,1)
start_road_2 = ("road-265524416",)
lane_idx_2 = (0,)
start_1 = product(start_road_1,lane_idx_1)
start_2 = product(start_road_2,lane_idx_2)
start = list(start_1) + list(start_2)
end_road = ("road-265524329","road-265524695","road-265543753-265524440")

route_comb = product(start,end_road)
leader_mission = []
ego_missions = []
for route in route_comb:
    leader_mission.append(
        Mission(Route(
            begin=(route[0][0],route[0][1],10),end=(route[1],0,"max")),
        )
    )
    ego_missions.append(
        EndlessMission(
            begin=(route[0][0],route[0][1],5), 
            entry_tactic=TrapEntryTactic(
                wait_to_hijack_limit_s=0,
                default_entry_speed=1,
            ),
        )
    )

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
        social_agent_missions={"leader":(leader_actor, leader_mission)},
        ego_missions=ego_missions,
        map_spec=t.MapSpec(source=f"{scenario_path}", lanepoint_spacing=1.0),
        # traffic_histories=traffic_histories,
    ),
    output_dir=Path(__file__).parent,
)
