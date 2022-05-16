from pathlib import Path
from smarts.sstudio.types import Distribution
from smarts.core import seed
from smarts.sstudio import gen_scenario
from smarts.sstudio import types as t
import random
import numpy as np
seed(42)

traffic = t.Traffic(
    flows=[
        t.Flow(
            route=t.Route(
                begin=("445633931",random.randint(0,2), 'random'),
                end=("445633932", random.randint(0,2), "max"),  
            ),
            rate=10*10,
            begin=np.random.exponential(scale=2.5),
            actors={t.TrafficActor(name="car",speed=Distribution(mean=0.5, sigma=0.8),vehicle_type=vehicle_type): 1},
        )
        for vehicle_type in [
            "passenger",
            "bus",
            "coach",
            "truck",
            "trailer",
        ]
    ]
)

laner_actor = t.SocialAgentActor(
    name="keep-lane-agent",
    agent_locator="zoo.policies:keep-lane-agent-v0",
)

gen_scenario(
    t.Scenario(
        traffic={"basic": traffic},
        social_agent_missions={
            "all": ([laner_actor], [t.Mission(route=t.RandomRoute())])
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
