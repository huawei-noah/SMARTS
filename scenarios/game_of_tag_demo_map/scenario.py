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
        begin=("top_left", 0, 15) # pred 1
    ),
    t.EndlessMission(
        begin=("top", 1, 5) # pred 2
    ),
    t.EndlessMission(
        begin=("top_left", 1, 15) # pred 3
    ),
    t.EndlessMission(
        begin=("top", 3, 30) # pred 4
    ),
    t.EndlessMission(
        begin=("top", 2, 30) # prey 1
    ),
    t.EndlessMission(
        begin=("top", 0, 5) # prey 2
    ),   
]


scenario = t.Scenario(
    # traffic={"all": traffic},
    ego_missions=ego_missions,
)

gen_scenario(scenario, output_dir=str(Path(__file__).parent))
