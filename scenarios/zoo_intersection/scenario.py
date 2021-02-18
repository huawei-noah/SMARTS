import os
import random

from smarts.sstudio import gen_missions, gen_social_agent_missions, gen_traffic
from smarts.sstudio.types import (
    Distribution,
    EndlessMission,
    Flow,
    LaneChangingModel,
    Mission,
    RandomRoute,
    Route,
    SocialAgentActor,
    Traffic,
    TrafficActor,
)

scenario = os.path.dirname(os.path.realpath(__file__))

# Traffic Vehicles
#
car = TrafficActor(
    name="car",
)

cooperative_car = TrafficActor(
    name="cooperative_car",
    speed=Distribution(sigma=0.2, mean=1.0),
    lane_changing_model=LaneChangingModel(impatience=0.1, cooperative=1),
)

aggressive_car = TrafficActor(
    name="aggressive_car",
    speed=Distribution(sigma=0.2, mean=1.0),
    lane_changing_model=LaneChangingModel(impatience=1, cooperative=0.1),
)

horizontal_routes = [("west-WE", "east-WE"), ("east-EW", "west-EW")]
turn_left_routes = [("east-EW", "south-NS")]
turn_right_routes = [("west-WE", "south-NS")]

for name, routes in {
    "horizontal": horizontal_routes,
    "turns": turn_left_routes + turn_right_routes,
}.items():
    traffic = Traffic(
        flows=[
            Flow(
                route=Route(
                    begin=(f"edge-{r[0]}", 0, "random"), end=(f"edge-{r[1]}", 0, "max")
                ),
                rate=random.randint(50, 100),
                actors={cooperative_car: 0.35, car: 0.20, aggressive_car: 0.45},
            )
            for r in routes
        ]
    )

    for seed in [0, 5]:
        gen_traffic(
            scenario,
            traffic,
            name=f"{name}-{seed}",
            seed=seed,
        )


# Social Agents
#
# N.B. You need to have the agent_locator in a location where the left side can be resolved
#   as a module in form:
#       "this.resolved.module:attribute"
#   In your own project you would place the prefabs script where python can reach it
social_agent1 = SocialAgentActor(
    name="zoo-car1",
    agent_locator="scenarios.zoo_intersection.agent_prefabs:zoo-agent2-v0",
    initial_speed=20,
)
social_agent2 = SocialAgentActor(
    name="zoo-car2",
    agent_locator="scenarios.zoo_intersection.agent_prefabs:zoo-agent2-v0",
    initial_speed=20,
)

gen_social_agent_missions(
    scenario,
    social_agent_actor=social_agent2,
    name=f"s-agent-{social_agent2.name}",
    missions=[Mission(RandomRoute())],
)

gen_social_agent_missions(
    scenario,
    social_agent_actor=social_agent1,
    name=f"s-agent-{social_agent1.name}",
    missions=[
        EndlessMission(begin=("edge-south-SN", 0, 30)),
        Mission(Route(begin=("edge-west-WE", 0, 10), end=("edge-east-WE", 0, 10))),
    ],
)

# Agent Missions
gen_missions(
    scenario=scenario,
    missions=[
        Mission(Route(begin=("edge-east-EW", 0, 10), end=("edge-south-NS", 0, 10))),
        Mission(Route(begin=("edge-south-SN", 0, 10), end=("edge-east-WE", 0, 10))),
    ],
)
