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

open_agent_actor = t.SocialAgentActor(
    name="open-agent", agent_locator="open_agent:open_agent-v0"
)

laner_actor = t.SocialAgentActor(
    name="keep-lane-agent", agent_locator="zoo.policies:keep-lane-agent-v0",
)

gen_scenario(
    t.Scenario(
        traffic={"basic": traffic},
        social_agent_missions={
            "all": ([laner_actor, open_agent_actor], [t.Mission(route=t.RandomRoute())])
        },
        bubbles=[
            t.Bubble(
                zone=t.PositionalZone(pos=(50, 0), size=(10, 15)),
                margin=5,
                actor=open_agent_actor,
                follow_actor_id=t.Bubble.to_actor_id(laner_actor, mission_group="all"),
                follow_offset=(-7, 10),
            ),
        ],
    ),
    output_dir=Path(__file__).parent,
)
