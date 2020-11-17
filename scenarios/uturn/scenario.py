import os
from pathlib import Path

from smarts.sstudio import gen_scenario
from smarts.sstudio import types as t


traffic = t.Traffic(
    flows=[
        t.Flow(
            route=t.RandomRoute(), rate=3600, actors={t.TrafficActor(name="car"): 1.0},
        )
    ]
)

social_agent_missions = {
    "all": (
        [
            t.SocialAgentActor(
                name="open-agent", agent_locator="open_agent:open_agent-v0"
            ),
        ],
        [
            t.Mission(
                t.Route(begin=("edge-west-WE", 0, 30), end=("edge-south-NS", 0, 40))
            )
        ],
    ),
}

gen_scenario(
    scenario=t.Scenario(
        traffic={"basic": traffic}, social_agent_missions=social_agent_missions,
    ),
    output_dir=Path(__file__).parent,
)
