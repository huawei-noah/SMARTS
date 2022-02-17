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

laner_agent = t.SocialAgentActor(
    name="keep-lane-agent",
    agent_locator="zoo.policies:keep-lane-agent-v0",
)

buddha_agent = t.SocialAgentActor(
    name="buddha-agent",
    agent_locator="scenarios.tests.multi_agents_loop.agent_prefabs:buddha-agent-v0",
)

gen_scenario(
    t.Scenario(
        traffic={"basic": traffic},
        social_agent_groups={
            "group_1": [
                (a, t.Mission(route=t.RandomRoute()))
                for a in [laner_agent, buddha_agent]
            ],
            "group_2": [
                (a, t.Mission(route=t.RandomRoute()))
                for a in [laner_agent, buddha_agent]
            ],
            "group_3": [
                (a, t.Mission(route=t.RandomRoute()))
                for a in [laner_agent, buddha_agent]
            ],
            "group_4": [
                (a, t.Mission(route=t.RandomRoute()))
                for a in [laner_agent, buddha_agent]
            ],
            "group_5": [
                (a, t.Mission(route=t.RandomRoute()))
                for a in [laner_agent, buddha_agent]
            ]
        },
    ),
    output_dir=Path(__file__).parent,
)
