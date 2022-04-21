from pathlib import Path

from smarts.sstudio import gen_scenario
from smarts.sstudio import types as t

traffic = t.Traffic(
    flows=[
        t.Flow(
            route=t.Route(
                begin=("E1", lane_idx, "random"),
                end=("E1", lane_idx, "max"),
            ),
            rate=3,
            actors={t.TrafficActor("car"): 1},
        )
        for lane_idx in range(3)
    for _ in range(3)
    ]
)

scenario = t.Scenario(
    traffic={"all": traffic},
)

gen_scenario(scenario, output_dir=str(Path(__file__).parent))
