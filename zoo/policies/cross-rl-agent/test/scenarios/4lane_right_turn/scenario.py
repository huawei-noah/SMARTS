from pathlib import Path

from smarts.sstudio.genscenario import gen_scenario
from smarts.sstudio.types import (
    Distribution,
    Flow,
    LaneChangingModel,
    Mission,
    RandomRoute,
    Route,
    Scenario,
    SocialAgentActor,
    Traffic,
    TrafficActor,
)

social_vehicle_num = 100

right_traffic_actor = TrafficActor(
    name="car",
    speed=Distribution(sigma=0.2, mean=1),
    lane_changing_model=LaneChangingModel(impatience=0),
)

social_agent_missions = {
    "all": (
        [
            SocialAgentActor(
                name="open-agent",
                agent_locator="cross_rl_agent:cross_rl_agent-v0",
                initial_speed=20,
            ),
        ],
        [
            Mission(
                route=Route(begin=("edge-south-SN", 0, 10), end=("edge-east-WE", 0, 8)),
            ),
        ],
    ),
}

scenario = Scenario(
    traffic={
        "basic": Traffic(
            flows=[
                Flow(
                    route=RandomRoute(),
                    rate=1,
                    actors={right_traffic_actor: 1.0},
                )
                for i in range(social_vehicle_num)
            ]
        )
    },
    social_agent_missions=social_agent_missions,
)

gen_scenario(scenario=scenario, output_dir=Path(__file__).parent, ovewrite=True)
