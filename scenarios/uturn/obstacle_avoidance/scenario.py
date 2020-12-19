from pathlib import Path

from smarts.sstudio import gen_missions, gen_traffic, gen_scenario
from smarts.sstudio.types import (
    Route,
    Mission,
    UTurn,
)
from smarts.sstudio import types as t


scenario = str(Path(__file__).parent)

traffic = t.Traffic(
    flows=[
        t.Flow(
            route=Route(begin=("gneE1", 0, 90), end=("gneE1", 0, "max")),
            rate=1,
            actors={t.TrafficActor("car", max_speed=50 / 3.6): 1},
        )
    ],
)


ego_missions = [
    Mission(Route(begin=("gneE2", 2, 330), end=("gneE1", 2, "max")), task=UTurn()),
]


social_agent_missions = {
    "all": (
        [
            t.SocialAgentActor(
                name="non-interactive",
                agent_locator="zoo.policies:non-interactive-agent-v0",
            ),
        ],
        [t.Mission(Route(begin=("gneE1", 0, 150), end=("gneE1", 0, "max")))],
    )
}


gen_scenario(
    scenario=t.Scenario(
        # traffic={"basic": traffic},
        ego_missions=ego_missions,
        social_agent_missions=social_agent_missions,
    ),
    output_dir=Path(__file__).parent,
)
