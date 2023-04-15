import random
from pathlib import Path

from smarts.sstudio import gen_scenario
from smarts.sstudio import types as t
from smarts.sstudio.types import (EndlessMission, TrapEntryTactic)

traffic = t.Traffic(
    flows=[
        t.Flow(
            route=t.Route(
                begin=(route[0], random.randint(0, 2), "random"),
                end=(route[1], random.randint(0, 2), "max"),
            ),
            repeat_route=True,
            rate=1,
            end=10,  # `rate=1` adds 1 additional vehicle per hour. So set `end` < 1*60*60 secs to avoid addition of more vehicles after the initial flow. This prevents traffic congestion.
            actors={
                t.TrafficActor(
                    name="car",
                    speed=t.Distribution(mean=0.5, sigma=0.8),
                    vehicle_type=random.choice(
                        ["passenger", "coach", "bus", "trailer", "truck"]
                    ),
                ): 1
            },
        )
        for route in [("445633931", "445633932"), ("445633932", "445633931")] * 12
    ],
    trips=[
        t.Trip("target", route=t.RandomRoute(), depart=0.5),
    ],
)

laner_actor = t.SocialAgentActor(
    name="keep-lane-agent",
    agent_locator="zoo.policies:keep-lane-agent-v0",
)

ego_missions = [
    EndlessMission(
        begin=("445633931", 0, 5),
        start_time=0,
        entry_tactic=TrapEntryTactic(wait_to_hijack_limit_s=0, default_entry_speed=10),
    ),
    EndlessMission(
        begin=("445633931", 1, 15),
        start_time=0,
        entry_tactic=TrapEntryTactic(wait_to_hijack_limit_s=0, default_entry_speed=10),
    ),
    EndlessMission(
        begin=("445633931", 2, 25),
        start_time=0,
        entry_tactic=TrapEntryTactic(wait_to_hijack_limit_s=0, default_entry_speed=10),
    ),
    EndlessMission(
        begin=("445633931", 2, 35),
        start_time=0,
        entry_tactic=TrapEntryTactic(wait_to_hijack_limit_s=0, default_entry_speed=10),
    ),
]

gen_scenario(
    t.Scenario(
        traffic={"basic": traffic},
        ego_missions=ego_missions,
        social_agent_missions={
            "all": (
                [laner_actor],
                [
                    t.Mission(
                        route=t.RandomRoute(),
                        entry_tactic=t.IdEntryTactic("target", patience=10),
                    )
                ],
            )
        },
        bubbles=[
            t.Bubble(
                zone=t.PositionalZone(pos=(50, 0), size=(10, 15)),
                margin=5,
                actor=laner_actor,
                follow_actor_id=t.Bubble.to_actor_id(laner_actor, mission_group="all"),
                follow_offset=(-7, 10),
            ),
        ],
    ),
    output_dir=Path(__file__).parent,
)
