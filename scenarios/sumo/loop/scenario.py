import random
from pathlib import Path

from smarts.core.colors import Colors
from smarts.sstudio import gen_scenario
from smarts.sstudio import types as t

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
        t.Trip("other_interest", route=t.RandomRoute(), depart=0.5),
        t.Trip(
            "leader",
            route=t.Route(begin=("445633931", 0, 0.2), end=("445633931", 0, 20)),
            depart=0,
        ),
    ],
)

laner_actor = t.SocialAgentActor(
    name="keep-lane-agent",
    agent_locator="zoo.policies:keep-lane-agent-v0",
)

gen_scenario(
    t.Scenario(
        traffic={"basic": traffic},
        social_agent_missions={
            "all": (
                [laner_actor],
                [
                    t.Mission(
                        route=t.RandomRoute(),
                        entry_tactic=t.IdEntryTactic(
                            start_time=0.1,
                            actor_id="other_interest",
                            condition=t.TimeWindowCondition(0.1, 20.0),
                        ),
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
        scenario_metadata=t.ScenarioMetadata(
            actor_of_interest_re_filter=r"^(leader)|(other_interest)$",
            actor_of_interest_color=Colors.Blue,
        ),
    ),
    output_dir=Path(__file__).parent,
)
