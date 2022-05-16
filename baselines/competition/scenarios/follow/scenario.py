import random
from pathlib import Path
import numpy as np
from smarts.sstudio.types import Distribution
from smarts.sstudio import gen_scenario
from smarts.sstudio import types as t

actors = [
    t.SocialAgentActor(
        name=f"keep-lane-agent_{num}",
        agent_locator="zoo.policies:keep-lane-agent-v0",
    )
    for num in range(1)
]

edges = ["gneE1", "gneE2", "gneE3", "gneE4"]
start_edge = random.choice(edges)
start_lane = random.randint(0, 2)
start_offset = random.randint(0, 60)  # road is 80m long, so leave space for the leader
distance = 10

flow_lead = [
    t.Flow(
        route=t.Route(
            begin=(start_edge, start_lane, start_offset + distance),
            end=("gneE4", 1, "max"),
        ),
        rate=1,
        actors={t.TrafficActor("leader", min_gap=Distribution(mean=2.5, sigma=0)): 1},
    )
]
flow_follow = [
    t.Flow(
        route=t.Route(
            begin=(start_edge, start_lane, start_offset),
            end=("gneE4", 1, "max"),
        ),
        rate=1,
        actors={t.TrafficActor("follower"): 1},
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

# traffic = t.Traffic(flows=flow_lead + flow_follow)
# ego_mission = [
#     t.Mission(t.Route(begin=("gneE1", 1, 1), end=("gneE4", 1, "max")), start_time=3)
# ]
# scenario = t.Scenario(
#     traffic={"all": traffic},
#     ego_missions=ego_mission,
# )
# gen_scenario(scenario, output_dir=str(Path(__file__).parent))

gen_scenario(
    t.Scenario(
        social_agent_missions={
            "group-1": (actors,[t.EndlessMission(begin=(start_edge,start_lane,start_offset+distance))]),
            "group-2": (actors,[t.EndlessMission(begin=(start_edge,start_lane,start_offset))]),
        },
    ),
    output_dir=Path(__file__).parent,
)
