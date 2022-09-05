import os
import random
from pathlib import Path
import numpy as np

from smarts.sstudio import gen_traffic, gen_missions, gen_social_agent_missions, gen_scenario
from smarts.sstudio.types import (
    Scenario,
    Traffic,
    Flow,
    Route,
    RandomRoute,
    TrafficActor,
    SocialAgentActor,
    Distribution,
    LaneChangingModel,
    JunctionModel,
    Mission,
    EndlessMission,
)

scenario = os.path.dirname(os.path.realpath(__file__))

start_routes = ["south-SN", "west-WE", "north-NS"]
end_routes = ["east-WE", "south-NS", "west-EW", "north-SN"]

# Traffic Flows
for seed in np.random.choice(1000, 20, replace=False):
    actors = {}

    for i in range(6):
        car = TrafficActor(
            name = f'car_type_{i+1}',
            speed=Distribution(mean=np.random.uniform(0.6, 1.0), sigma=0.1),
            min_gap=Distribution(mean=np.random.uniform(2, 4), sigma=0.1),
            imperfection=Distribution(mean=np.random.uniform(0.3, 0.7), sigma=0.1),
            lane_changing_model=LaneChangingModel(impatience=np.random.uniform(0, 1.0), cooperative=np.random.uniform(0, 1.0)),
            junction_model=JunctionModel(ignore_foe_prob=np.random.uniform(0, 1.0), impatience=np.random.uniform(0, 1.0)),
        )

        actors[car] = 1/6

    flows = []

    for i, start in enumerate(start_routes):
        for end in end_routes:
            if end.split('-')[0] != start.split('-')[0]:
                flows.append(Flow(route=Route(begin=('edge-'+start, 0, "random"), end=('edge-'+end, 0, "random")), rate=100, actors=actors))

    flows.append(Flow(route=Route(begin=('edge-east-EN', 1, "random"), end=('edge-south-NS', 0, "random")), rate=100, actors=actors))
    #flows.append(Flow(route=Route(begin=('edge-east-EN', 1, "random"), end=('edge-north-SN', 0, "random")), rate=100, actors=actors))
    #flows.append(Flow(route=Route(begin=('edge-east-EN', 1, "random"), end=('edge-west-EW',  0, "random")), rate=100, actors=actors))

    traffic = Traffic(flows=flows)
    gen_traffic(scenario, traffic, seed=seed, name=f'traffic_{seed}')

# Agent Missions
gen_missions(scenario=scenario, missions=[Mission(Route(begin=("edge-east-EW", 0, 1), end=("edge-south-NS", 0, "max")), start_time=30)])
