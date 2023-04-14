from itertools import product
from pathlib import Path

from smarts.sstudio.genscenario import gen_scenario
from smarts.sstudio.types import (
    EndlessMission,
    MapSpec,
    Scenario,
    SocialAgentActor,
    TrapEntryTactic,
    Via,
)

first_via = (0, 1)
second_via = (0, 1)
third_via = (0, 1)

route_comb = product(first_via, second_via, third_via)

leader_mission = []
for route in route_comb:
    leader_mission.append(
        EndlessMission(
            begin=("E0", 1, 20),
            via=(
                Via(
                    "E0",
                    lane_offset=30,
                    lane_index=1,
                    required_speed=10,
                ),
                Via(
                    "E0",
                    lane_offset=100,
                    lane_index=route[0],
                    required_speed=20,
                ),
                Via(
                    "E0",
                    lane_offset=170,
                    lane_index=route[1],
                    required_speed=13,
                ),
                Via(
                    "E0",
                    lane_offset=240,
                    lane_index=route[2],
                    required_speed=13,
                ),
            ),
        ),
    )

leader_actor = [
    SocialAgentActor(
        name="Leader-007",
        agent_locator="zoo.policies:chase-via-points-agent-v0",
        initial_speed=0,
    )
]

ego_missions = [
    EndlessMission(
        begin=("E0", 1, 5),
        entry_tactic=TrapEntryTactic(wait_to_hijack_limit_s=0, default_entry_speed=0),
    )
]

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
