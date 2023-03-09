from itertools import product
from pathlib import Path

from smarts.sstudio.genscenario import gen_scenario
from smarts.sstudio.types import (
    EndlessMission,
    MapSpec,
    Scenario,
    SocialAgentActor,
    Via,
)

begin_road = ["E_start", "E_on"]
first_via = (0, 1, 2)
second_via = (0, 1, 2)
third_via = (0, 1, 2)

route_comb = product(begin_road, first_via, second_via, third_via)
leader_mission = []
for route in route_comb:
    leader_mission.append(
        EndlessMission(
            begin=(route[0], 2, 20),
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
                    "E_off",
                    lane_offset=50,
                    lane_index=0,
                    required_speed=7,
                ),
            ),
        )
    )

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
