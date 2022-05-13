from pathlib import Path

import numpy as np
import random
from smarts.sstudio import gen_scenario
from smarts.sstudio import types as t
from smarts.sstudio.types import Distribution


flow_lead = [
    t.Flow(
        route=t.Route(
            begin=("gneE1",1,10),
            end=("gneE4",1,'max'),
        ),
        rate=1,
        actors={t.TrafficActor("car"): 1},
    )
]
flow_follow = [
    t.Flow(
        route=t.Route(
            begin=("gneE1",1,1),
            end=("gneE4",1,'max'),
        ),
        rate=1,
        actors={t.TrafficActor("car"): 1},
    )
]
# traffic = t.Traffic(
#     flows=[
#         t.Flow(
#             route=t.Route(
#                 begin=("gneE1",1, 'random'),
#                 end=("gneE4",1, "max"),
#             ),
#             rate=2,
#             # begin=np.random.exponential(scale=2.5),
#             actors={t.TrafficActor("car",speed=Distribution(mean=0.5, sigma=0.8)): 1},
#         )
#         for _ in range(3)
#     for _ in range(2)
#     ]
# )

traffic = t.Traffic(flows=flow_lead+flow_follow)

ego_mission = [t.Mission(t.Route(begin=("gneE1",1,1),end=("gneE4",1,'max')),start_time=3)]

scenario = t.Scenario(
    traffic={"all": traffic},
    ego_missions=ego_mission,
)

gen_scenario(scenario, output_dir=str(Path(__file__).parent))
