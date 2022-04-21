from pathlib import Path

from smarts.sstudio import gen_scenario
from smarts.sstudio import types as t

flow1 = [t.Flow(
            route=t.Route(
                begin=("-E3",0, "random"), end=("-E0",0, "max")
            ),
            rate=1,
            actors={t.TrafficActor("car"): 1},
            
        )
        for _ in range(1)]

flow2 = [t.Flow(
            route=t.Route(
                begin=("E0",0, "random"), end=("E3",0, "max")
            ),
            rate=1,
            actors={t.TrafficActor("car"): 1},
            
        )
        for _ in range(1)]

flow3 = [t.Flow(
            route=t.Route(
                begin=("E4", lane, "random"), end=("E1", lane, "max")
            ),
            rate=1,
            actors={t.TrafficActor("car"): 1},
            
        ) for lane in range(2)
        for _ in range(2)]

flow4 = [t.Flow(
            route=t.Route(
                begin=("-E1", lane, "random"), end=("-E4", lane, "max")
            ),
            rate=1,
            actors={t.TrafficActor("car"): 1},
            
        ) for lane in range(2)
        for _ in range(2)]

flow5 = [t.Flow(
            route=t.Route(
                begin=("E0", 0, "random"), end=("E1", 1, "max")
            ),
            rate=1,
            actors={t.TrafficActor("car"): 1},
            
        )
        for _ in range(1)]


traffic = t.Traffic(
    flows=flow1+flow2+flow3+flow4+flow5
)

gen_scenario(
    scenario=t.Scenario(
        traffic={"basic": traffic},
    ),
    output_dir=Path(__file__).parent,
)
