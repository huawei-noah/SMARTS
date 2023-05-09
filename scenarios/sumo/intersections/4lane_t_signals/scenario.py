from pathlib import Path

from smarts.core.colors import Colors
from smarts.sstudio import gen_scenario
from smarts.sstudio import types as t

traffic = t.Traffic(
    flows=[
        t.Flow(
            route=t.Route(begin=("edge-west-WE", 0, 10), end=("edge-east-WE", 0, "max")),
            repeat_route=True,
            rate=400,
            actors={t.TrafficActor("car"): 1},
        ),
        t.Flow(
            route=t.Route(begin=("edge-east-EW", 0, 10), end=("edge-west-EW", 0, "max")),
            repeat_route=True,
            rate=400,
            actors={t.TrafficActor("car"): 1},
        ),
    ]
)

ego_missions = [
    t.EndlessMission(
        begin=("edge-south-SN", 1, 20),
    )
]

gen_scenario(
    scenario=t.Scenario(
        traffic={"basic": traffic},
        ego_missions=ego_missions,
        scenario_metadata=t.ScenarioMetadata(r".*-1.*", Colors.Yellow),
    ),
    output_dir=Path(__file__).parent,
)
