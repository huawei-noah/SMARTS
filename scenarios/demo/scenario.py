from pathlib import Path

from smarts.sstudio import gen_scenario
from smarts.sstudio import types as t

traffic = t.Traffic(
    flows=[
        t.Flow(
            route=t.RandomRoute(),
            rate=1,
            actors={t.TrafficActor("car", max_speed=8): 1},
            begin=2,
        )
    ]
)


ego_missions = [
    t.Mission(
        t.Route(begin=("-gneE72", 1, 5), end=("-gneE71", 0, "max")),
        task=t.CutIn(),
    )
]

scenario = t.Scenario(
    traffic={"all": traffic},
    # ego_missions=ego_missions,
)

gen_scenario(scenario, output_dir=str(Path(__file__).parent))
