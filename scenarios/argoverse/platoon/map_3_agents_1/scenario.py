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
scenario_id = "c624608b-fd20-43c9-bc2c-c4181ce9dafa"
scenario_path = Path(PATH) / scenario_id

end_road = (
    ("road-353613894-353613949", 1),
    ("road-353635854", 0),
    ("road-353612658", 0),
    ("road-353612841", 0),
)

route_comb = end_road
leader_mission = []
for route in route_comb:
    leader_mission.append(
        Mission(
            Route(
                begin=("road-353614080-353614150", 0, 10),
                end=(route[0], route[1], "max"),
            ),
        )
    )

ego_missions = [
    EndlessMission(
        begin=("road-353614080-353614150", 0, 5),
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
