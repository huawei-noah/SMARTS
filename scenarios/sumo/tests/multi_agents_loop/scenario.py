from pathlib import Path

from smarts.sstudio import gen_scenario
from smarts.sstudio import types as t

traffic = t.Traffic(
    flows=[
        t.Flow(
            route=t.RandomRoute(),
            repeat_route=True,
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


def make_laner(id_):
    return t.SocialAgentActor(
        name=f"keep-lane-agent_{id_}",
        agent_locator="zoo.policies:keep-lane-agent-v0",
    )


def make_buddha(id_):
    return t.SocialAgentActor(
        name=f"buddha-actor_{id_}",
        agent_locator="scenarios.sumo.tests.multi_agents_loop.agent_prefabs:buddha-agent-v0",
    )


gen_scenario(
    t.Scenario(
        traffic={"basic": traffic},
        social_agent_missions={
            f"group_{i}": (
                [make_laner(i), make_buddha(i)],
                [t.Mission(route=t.RandomRoute())],
            )
            for i in range(1, 6)
        },
    ),
    output_dir=Path(__file__).parent,
)
