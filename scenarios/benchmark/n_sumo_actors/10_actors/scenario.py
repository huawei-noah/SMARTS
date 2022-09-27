
from pathlib import Path

from smarts.sstudio import gen_scenario
from smarts.sstudio import types as t


traffic = t.Traffic(
    flows=[
        t.Flow(
            route=t.Route(
                begin=("gneE01", lane_idx, 0),
                end=("gneE01.132", lane_idx, "max"),
            ),
            rate=3600,
            end=2,
            actors={t.TrafficActor("car"): 1},
        )
        for lane_idx in range(5)
    ]
)

scenario = t.Scenario(
    traffic={"all": traffic},
)

gen_scenario(scenario, output_dir=str(Path(__file__).parent))