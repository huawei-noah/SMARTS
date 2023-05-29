import os
from pathlib import Path

from smarts.sstudio import gen_scenario
from smarts.sstudio.types import Mission, Route, Scenario, SocialAgentActor, Traffic
from smarts.sstudio.types.entry_tactic import IdEntryTactic
from smarts.sstudio.types.route import RandomRoute
from smarts.sstudio.types.traffic import Trip


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
        ego_missions=[
            Mission(
                Route(begin=("edge-north-NS", 0, 20), end=("edge-south-NS", 0, "max")),
                entry_tactic=IdEntryTactic(2, "other_interest"),
            )
        ],
        social_agent_missions={
            "group-1": (actor_gen(1), [to_mission("edge-north-NS", "edge-south-NS")]),
            "group-2": (actor_gen(2), [to_mission("edge-west-WE", "edge-east-WE")]),
            "group-3": (actor_gen(3), [to_mission("edge-east-EW", "edge-west-EW")]),
            "group-4": (actor_gen(4), [to_mission("edge-south-SN", "edge-north-SN")]),
        },
        traffic={
            "basic": Traffic(
                flows=[],
                trips=[
                    Trip(
                        "other_interest",
                        route=Route(
                            begin=("edge-north-NS", 0, 20),
                            end=("edge-south-NS", 0, "max"),
                        ),
                        depart=1,
                    ),
                ],
            )
        },
    ),
    output_dir=Path(__file__).parent,
)
