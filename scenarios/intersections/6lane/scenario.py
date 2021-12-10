import os
import random
from pathlib import Path
from typing import Sequence

from smarts.sstudio import gen_scenario
from smarts.sstudio.types import Mission, Route, Scenario, SocialAgentActor

speeds = [10, 30, 80]


def to_mission(start_edge, end_edge):
    route = Route(begin=(start_edge, 1, 0), end=(end_edge, 1, "max"))
    return Mission(route=route)


missions: Sequence[Mission] = [
    to_mission(*edge_pairs)
    for edge_pairs in [
        ("edge-north-NS", "edge-south-NS"),
        ("edge-west-WE", "edge-east-WE"),
        ("edge-east-EW", "edge-west-EW"),
        ("edge-south-SN", "edge-north-SN"),
    ]
]

# Make random deterministic
random.seed(42)
actor_mission_pair_groups = {}
for i in range(4):
    group = f"group{i+1}"
    actor_mission_pairs = []
    for mission in missions:
        speed = speeds[random.randint(0, len(speeds) - 1)]
        route_begin = mission.route.begin[0]
        actor_mission_pairs.append(
            (
                SocialAgentActor(
                    name=f"non-interactive-agent-{speed}-{group}-v0-{route_begin}",
                    agent_locator="zoo.policies:non-interactive-agent-v0",
                    policy_kwargs={"speed": speed},
                ),
                mission,
            )
        )
    actor_mission_pair_groups[group] = actor_mission_pairs

gen_scenario(
    Scenario(social_agents=actor_mission_pair_groups),
    output_dir=Path(__file__).parent,
)
