import os
from pathlib import Path

from smarts.sstudio import gen_scenario
from smarts.sstudio import types as t

traffic = t.Traffic(
    flows=[
        t.Flow(
            route=t.Route(
                begin=("edge-west-EW", 0, 10), end=("edge-west-EW", 0, "max")
            ),
            rate=1,
            actors={t.TrafficActor(name="car", max_speed=30 / 3.6): 1.0},
            begin=2,
        )
    ]
)


social_agent_missions = {
    "all": (
        [
            t.SocialAgentActor(
                name="open-agent-default",
                agent_locator="open_agent:open_agent-v0",
                # policy_kwargs={"gains": stable_config},
            ),
        ],
        [
            t.Mission(
                t.Route(begin=("edge-west-WE", 0, 10), end=("edge-west-EW", 0, "max")),
                task=t.UTurn(target_lane_index=0),
            ),
        ],
    ),
}

ego_missions = [
    t.Mission(
        t.Route(begin=("edge-west-WE", 0, 10), end=("edge-west-EW", 0, "max")),
        task=t.UTurn(),
    )
]

gen_scenario(
    scenario=t.Scenario(
        traffic={"basic": traffic},
        ego_missions=ego_missions
        # social_agent_missions=social_agent_missions,
    ),
    output_dir=Path(__file__).parent,
)
