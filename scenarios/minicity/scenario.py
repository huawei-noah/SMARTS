import random
from pathlib import Path

from smarts.sstudio import types as t
from smarts.sstudio import gen_scenario

NUM_TRAFFIC_FLOWS = 500

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
        for _ in range(NUM_TRAFFIC_FLOWS)
    ]
)

open_agent_actor = t.SocialAgentActor(
    name="open-agent", agent_locator="open_agent:open_agent-v0"
)

laner_actor = t.SocialAgentActor(
    name="keep-lane-agent", agent_locator="zoo.policies:keep-lane-agent-v0",
)

travelling_bubbles = [
    t.Bubble(
        zone=t.PositionalZone(pos=(50, 0), size=(10, 50)),
        margin=5,
        actor=open_agent_actor,
        follow_actor_id=t.Bubble.to_actor_id(laner_actor, mission_group="all"),
        follow_offset=(-7, 10),
    )
]

static_bubbles = [
    t.Bubble(zone=t.MapZone((id_, 0, 10), 200, 1), margin=5, actor=laner_actor,)
    for id_ in [
        "21675239",
        "126742590#1",
        "-77720372",
        "-263506114#6",
        "-33002812#1",
        "14632489#1",
    ]
]

gen_scenario(
    t.Scenario(
        traffic={"basic": traffic},
        social_agent_missions={
            "all": ([laner_actor, open_agent_actor], [t.Mission(route=t.RandomRoute())])
        },
        bubbles=[*travelling_bubbles, *static_bubbles],
    ),
    output_dir=Path(__file__).parent,
)
