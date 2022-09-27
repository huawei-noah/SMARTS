import os
from pathlib import Path

from smarts.sstudio import gen_scenario
from smarts.sstudio.types import Mission, Route, Scenario, SocialAgentActor,RandomRoute

actors = [
    SocialAgentActor(
        name=f"non-interactive-agent-{speed}-v0",
        agent_locator="zoo.policies:non-interactive-agent-v0",
        policy_kwargs={"speed": speed},
    )
    for speed in [10, 30, 80]
]


def to_missions(agent_num):
    missions = {}
    for i in range(0,agent_num):
        missions[f'group-{i}']=tuple((actors, [Mission(route=RandomRoute())]),)
    return missions

gen_scenario(
    Scenario(
        social_agent_missions=to_missions(1)
    ),
    output_dir=Path(__file__).parent,
)
