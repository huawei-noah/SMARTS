
from pathlib import Path

from smarts.sstudio import gen_scenario
from smarts.sstudio import types as t


traffic = t.Traffic(
    flows=[
        t.Flow(
            route=t.Route(
                begin=("gneE01", 0, 0),
                end=("gneE01.132", 0, "max"),
            ),
            rate=1,
            actors={t.TrafficActor("car"): 1},
        )
    ]
)

scenario = t.Scenario(
    traffic={"all": traffic},
)

gen_scenario(scenario, output_dir=str(Path(__file__).parent))