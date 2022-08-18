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
    t.EndlessMission(begin=("top", 2, 5)),  # pred
    t.EndlessMission(begin=("top", 2, 30)),  # prey
]


scenario = t.Scenario(
    # traffic={"all": traffic},
    ego_missions=ego_missions,
)

gen_scenario(scenario, output_dir=str(Path(__file__).parent))
