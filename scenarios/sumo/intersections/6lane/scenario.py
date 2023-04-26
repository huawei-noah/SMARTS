import os
from pathlib import Path

from smarts.sstudio import gen_scenario
from smarts.sstudio.types import Mission, Route, Scenario, SocialAgentActor


def actor_gen(id_):
    return [
        SocialAgentActor(
            name=f"{id_}-non-interactive-agent-{speed}-v0",
            agent_locator="zoo.policies:non-interactive-agent-v0",
            policy_kwargs={"speed": speed},
        )
        for speed in [10, 30, 80]
    ]


def to_mission(start_edge, end_edge):
    route = Route(begin=(start_edge, 1, 0), end=(end_edge, 1, "max"))
    return Mission(route=route)


gen_scenario(
    Scenario(
        social_agent_missions={
            "group-1": (actor_gen(1), [to_mission("edge-north-NS", "edge-south-NS")]),
            "group-2": (actor_gen(2), [to_mission("edge-west-WE", "edge-east-WE")]),
            "group-3": (actor_gen(3), [to_mission("edge-east-EW", "edge-west-EW")]),
            "group-4": (actor_gen(4), [to_mission("edge-south-SN", "edge-north-SN")]),
        }
    ),
    output_dir=Path(__file__).parent,
)
