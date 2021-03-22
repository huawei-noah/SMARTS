import random
from pathlib import Path

from smarts.sstudio import gen_scenario
from smarts.sstudio import types as t

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
        for i in range(5)
    ]
)


scenario = t.Scenario(
    traffic={"all": traffic},
)

gen_scenario(scenario, output_dir=str(Path(__file__).parent))
