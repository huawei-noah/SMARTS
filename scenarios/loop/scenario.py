import random
from pathlib import Path

from smarts.sstudio import types as t
from smarts.sstudio import gen_scenario

traffic = t.Traffic(
    flows=[
        t.Flow(
            route=t.RandomRoute(),
            rate=60 * 60,
            actors={
                t.TrafficActor(
                    name="car",
                    vehicle_type=random.choice(
                        ["passenger", "bus", "coach", "truck", "trailer"]
                    ),
                ): 1
            },
        )
        for i in range(5)
    ]
)

actor = t.SocialAgentActor(
    name="keep-lane-agent", agent_locator="zoo.policies:keep-lane-agent-v0",
)

gen_scenario(
    t.Scenario(
        traffic={"basic": traffic},
        social_agent_missions={
            "all": ([actor], [t.Mission(route=t.RandomRoute())]),
        },
        bubbles=[
            t.Bubble(
                zone=t.PositionalZone(pos=(50, 0), size=(10, 15)),
                margin=5,
                actor=actor,
                # TODO: We either need a better API around ID or or use something
                #       other than ID...
                pinned_actor_id="social-agent-all-keep-lane-agent",
                pinned_offset=(-7, 10)
            ),
        ],
    ),
    output_dir=Path(__file__).parent
)
