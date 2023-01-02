from pathlib import Path

from smarts.sstudio import types as t
from smarts.sstudio import gen_scenario

traffic_actor = t.TrafficActor(
    name="car",
    speed=t.Distribution(sigma=0.2, mean=0.8),
)
traffic = t.Traffic(
    engine="SUMO",
    flows=[
        t.Flow(
            route=t.RandomRoute(),
            begin=0,
            end=10 * 60 * 60,  # Flow lasts for 10 hours.
            rate=50,
            actors={traffic_actor: 1},
        )
        for i in range(10)
    ],
)

gen_scenario(
    scenario=t.Scenario(
        traffic={"basic": traffic},
    ),
    output_dir=Path(__file__).parent,
)
