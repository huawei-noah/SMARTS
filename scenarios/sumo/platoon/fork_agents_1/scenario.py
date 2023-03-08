from pathlib import Path
from itertools import product
from smarts.sstudio.genscenario import gen_scenario
from smarts.sstudio.types import (
    MapSpec,
    EndlessMission,
    Route,
    Scenario,
    Via,
    SocialAgentActor,
    Mission,
)

end_road = ["E_left","E_right"]
begin_lane_idx = (0,1)
first_via = (0,1)
second_via = (0,1)
third_via = (0,1)
fourth_via = (0,1)

route_comb = product(begin_lane_idx,first_via,second_via,third_via,fourth_via,end_road)

leader_mission = []
for route in route_comb:
    leader_mission.append(
        EndlessMission(
            begin=("E0", route[0], 20),
            via=(
                Via(
                    "E0",
                    lane_offset=50,
                    lane_index=route[1],
                    required_speed=15,
                ),
                Via(
                    "E0",
                    lane_offset=100,
                    lane_index=route[2],
                    required_speed=20,
                ),
                Via(
                    "E0",
                    lane_offset=150,
                    lane_index=route[3],
                    required_speed=7,
                ),
                Via(
                    route[5],
                    lane_offset=50,
                    lane_index=route[4],
                    required_speed=7,
                ),
            ),
        )
    )
    print(leader_mission)


leader_actor = [
    SocialAgentActor(
        name="Leader-007",
        agent_locator="zoo.policies:chase-via-points-agent-v0",
    )
]

ego_missions = [EndlessMission(begin=("E0", 0, 5))]

scenario = Scenario(
    ego_missions=ego_missions,
    social_agent_missions={"leader": (leader_actor, leader_mission)},
    map_spec=MapSpec(
        source=Path(__file__).parent.absolute(),
        lanepoint_spacing=1.0,
    ),
)

gen_scenario(
    scenario=scenario,
    output_dir=Path(__file__).parent,
)
