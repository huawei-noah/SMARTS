import os
from pathlib import Path

from smarts.sstudio import gen_scenario
from smarts.sstudio import types as t


traffic = t.Traffic(
    flows=[
        t.Flow(
            route=t.Route(begin=("edge-west-EW", 0, 0), end=("edge-west-EW", 0, "max")),
            rate=1,
            actors={t.TrafficActor(name="car"): 1.0},
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
                t.Route(begin=("edge-west-WE", 0, 50), end=("edge-west-EW", 0, "max")),
                task=t.UTurn(target_lane_index=0),
            )
        ],
    ),
}

ego_missions = [
    t.Mission(
        t.Route(begin=("edge-west-WE", 0, 50), end=("edge-west-EW", 0, "max")),
        task=t.UTurn(),
    )
]

gen_scenario(
    scenario=t.Scenario(
        traffic={"basic": traffic},
        # ego_missions=ego_missions,
        social_agent_missions=social_agent_missions,
    ),
    output_dir=Path(__file__).parent,
)
