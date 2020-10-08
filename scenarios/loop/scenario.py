import random
from pathlib import Path

from smarts.sstudio import types as t
from smarts.sstudio import gen_scenario

traffic = t.Traffic(
    flows=[
        t.Flow(
            route=t.RandomRoute(),
            rate=60 * 60,
            actors={
                t.TrafficActor(
                    name="car",
                    vehicle_type=random.choice(
                        ["passenger", "bus", "coach", "truck", "trailer"]
                    ),
                ): 1
            },
        )
        for i in range(10)
    ]
)
gen_scenario(t.Scenario(traffic={"basic": traffic}), output_dir=Path(__file__).parent)
