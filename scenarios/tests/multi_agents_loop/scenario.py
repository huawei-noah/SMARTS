import random
from pathlib import Path

from smarts.sstudio import gen_scenario
from smarts.sstudio import types as t

traffic = t.Traffic(
    flows=[
        t.Flow(
            route=t.RandomRoute(),
            rate=60 * 60,
            actors={t.TrafficActor(name="car", vehicle_type=vehicle_type): 1},
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
            "group_1": (
                [laner_actor, laner_actor],
                [t.Mission(route=t.RandomRoute())],
            ),
            "group_2": (
                [laner_actor, laner_actor],
                [t.Mission(route=t.RandomRoute())],
            ),
            "group_3": (
                [laner_actor, laner_actor],
                [t.Mission(route=t.RandomRoute())],
            ),
            "group_4": (
                [laner_actor, laner_actor],
                [t.Mission(route=t.RandomRoute())],
            ),
            "group_5": (
                [laner_actor, laner_actor],
                [t.Mission(route=t.RandomRoute())],
            ),
        },
    ),
    output_dir=Path(__file__).parent,
)
