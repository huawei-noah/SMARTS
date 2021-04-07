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

# training missions
ego_missions = [
    t.EndlessMission(
        begin=("-gneE72", 1, 0) # pred 1
    ),
    t.EndlessMission(
        begin=("gneE71", 2, 0) # pred 2
    ),
    t.EndlessMission(
        begin=("gneE72", 2, 10) # pred 3
    ),
    t.EndlessMission(
        begin=("-gneE71",1, 5) # pred 4
    ),
    t.EndlessMission(
        begin=("-gneE72", 1, 10) # prey 1
    ),
    t.EndlessMission(
        begin=("gneE71", 0, 10) # prey 2
    ),   
]

# simplified mission for testing, 1 prey and 1 pred
# ego_missions = [
#     t.EndlessMission(
#         begin=("gneE71", 2, 0) # pred 1
#     ),
#     t.EndlessMission(
#         begin=("gneE71", 2, 30) # pred 2
#     )
# ]

scenario = t.Scenario(
    # traffic={"all": traffic},
    ego_missions=ego_missions,
)

gen_scenario(scenario, output_dir=str(Path(__file__).parent))
