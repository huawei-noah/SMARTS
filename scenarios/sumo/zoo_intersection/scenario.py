import random
from pathlib import Path

from smarts.sstudio import gen_scenario
from smarts.sstudio.types import (
    Distribution,
    EndlessMission,
    Flow,
    LaneChangingModel,
    Mission,
    Route,
    Scenario,
    SocialAgentActor,
    Traffic,
    TrafficActor,
)

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

traffic = {}
for name, routes in {
    "horizontal": horizontal_routes,
    "turns": turn_left_routes + turn_right_routes,
}.items():
    traffic[name] = Traffic(
        flows=[
            Flow(
                route=Route(
                    begin=(f"edge-{r[0]}", 0, "random"), end=(f"edge-{r[1]}", 0, "max")
                ),
                repeat_route=True,
                rate=random.randint(50, 100),
                actors={cooperative_car: 0.35, car: 0.20, aggressive_car: 0.45},
            )
            for r in routes
        ]
    )


# Social Agents
#
# N.B. You need to have the agent_locator in a location where the left side can be resolved
#   as a module in form:
#       "this.resolved.module:attribute"
#   In your own project you would place the prefabs script where python can reach it
social_agent1 = SocialAgentActor(
    name="zoo-car1",
    agent_locator="scenarios.sumo.zoo_intersection.agent_prefabs:zoo-agent2-v0",
    initial_speed=20,
)
social_agent2 = SocialAgentActor(
    name="zoo-car2",
    agent_locator="scenarios.sumo.zoo_intersection.agent_prefabs:zoo-agent2-v0",
    initial_speed=20,
)


gen_scenario(
    scenario=Scenario(
        traffic=traffic,
        ego_missions=[
            Mission(Route(begin=("edge-east-EW", 0, 10), end=("edge-south-NS", 0, 10))),
            Mission(Route(begin=("edge-south-SN", 0, 10), end=("edge-east-WE", 0, 10))),
        ],
        social_agent_missions={
            f"s-agent-{social_agent2.name}": (
                [social_agent2],
                [
                    Mission(
                        Route(
                            begin=("edge-south-SN", 0, 30), end=("edge-east-WE", 0, 10)
                        ),
                    ),
                ],
            ),
            f"s-agent-{social_agent1.name}": (
                [social_agent1],
                [
                    EndlessMission(begin=("edge-south-SN", 0, 10), start_time=0.7),
                ],
            ),
        },
    ),
    output_dir=Path(__file__).parent,
)
