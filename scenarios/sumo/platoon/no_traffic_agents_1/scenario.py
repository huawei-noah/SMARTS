from pathlib import Path

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


ego_missions = [EndlessMission(begin=("E0", 1, 5), start_time=0.5)]


leader_mission = [
    Mission(
        route=Route(
            begin=("E0", 1, 20),
            end=("E1", 0, "max"),
        ),
        via=(
            Via(
                "E0",
                lane_offset=40,
                lane_index=1,
                required_speed=15,
            ),
            Via(
                "E0",
                lane_offset=70,
                lane_index=0,
                required_speed=20,
            ),
            Via(
                "E1",
                lane_offset=50,
                lane_index=1,
                required_speed=7,
            ),
        ),
    ),
]


leader_actor = [
    SocialAgentActor(
        name="Agent_007",
        agent_locator="zoo.policies:chase-via-points-agent-v0",
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
