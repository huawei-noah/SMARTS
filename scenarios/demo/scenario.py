import random
from pathlib import Path

from smarts.sstudio import gen_scenario
from smarts.sstudio import types as t
from smarts.core import seed

seed(42)

# traffic = t.Traffic(
#     flows=[
#         t.Flow(
#             route=t.Route(
#                 begin=("-gneE69", 0, 10),
#                 end=("gneE77", 0, 0),
#             ),
#             rate=60*60,
#             actors={
#                 t.TrafficActor(
#                     name="car",
#                     vehicle_type=random.choice(
#                         ["passenger", "bus", "coach", "truck", "trailer"]
#                     ),
#                 ): 1
#             },
#         )
#     ]
# )

ego_missions = [
    t.Mission(
        route=t.Route(begin=("-gneE72", 1, 0), end=("gneE72", 0, 'max')), # pred 1
    ),
    t.Mission(
        route=t.Route(begin=("gneE71", 2, 0), end=("gneE72", 0, 'max')), # pred 2
    ),
    t.Mission(
        route=t.Route(begin=("-gneE72", 1, 10), end=("gneE72", 0, 'max')), # prey 1
    ),
    t.Mission(
        route=t.Route(begin=("gneE71", 0, 10), end=("gneE72", 0, 'max')), # prey 2
    ),
]

scenario = t.Scenario(
    # traffic={"all": traffic},
    ego_missions=ego_missions,
)

gen_scenario(scenario, output_dir=str(Path(__file__).parent))
