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

# begin_road_lane = [("E0", 1), ("E0", 2), ("E1", 0), ("E1", 1)]
# first_via = (0, 1, 2)
# second_via = (0, 1, 2)
# third_via = (1, 2)

begin_road_lane = [("E0", 1),]
first_via = (0, 1, 2)
second_via = (0, 1, 2)
third_via = (1, 2)
# This part should be reverted.

route_comb = product(begin_road_lane, first_via, second_via, third_via)

leader_mission = []
for route in route_comb:
    leader_mission.append(
        EndlessMission(
            begin=(route[0][0], route[0][1], 20),
            via=(
                Via(
                    "E2",
                    lane_offset=10,
                    lane_index=route[1],
                    required_speed=10,
                ),
                Via(
                    "E2",
                    lane_offset=30,
                    lane_index=route[2],
                    required_speed=18,
                ),
                Via(
                    "E2",
                    lane_offset=140,
                    lane_index=route[3],
                    required_speed=13,
                ),
                Via(
                    "E4",
                    lane_offset=40,
                    lane_index=0,
                    required_speed=15,
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
