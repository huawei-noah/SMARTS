from pathlib import Path

from smarts.sstudio import gen_scenario
from smarts.sstudio import types as t
from smarts.sstudio.types import Via

traffic = t.Traffic(
    flows=[
        t.Flow(
            route=t.Route(
                begin=("edge-west-WE", 0, 10), end=("edge-east-WE", 0, "max")
            ),
            rate=400,
            actors={t.TrafficActor("car"): 1},
        ),
        t.Flow(
            route=t.Route(
                begin=("edge-east-EW", 0, 10), end=("edge-west-EW", 0, "max")
            ),
            rate=400,
            actors={t.TrafficActor("car"): 1},
        ),
    ]
)

agent_prefabs = "scenarios.intersections.4lane_t.agent_prefabs"
bubbles = [
    t.Bubble(
        zone=t.MapZone(start=("edge-west-WE", 0, 50), length=10, n_lanes=1),
        margin=2,
        actor=t.SocialAgentActor(
            name="zoo-agent", agent_locator=f"{agent_prefabs}:zoo-agent-v0",
        ),
    ),
    t.Bubble(
        zone=t.PositionalZone(pos=(100, 100), size=(20, 20)),
        margin=2,
        actor=t.SocialAgentActor(
            name="motion-planner-agent",
            agent_locator=f"{agent_prefabs}:motion-planner-agent-v0",
        ),
    ),
]

ego_missions = [
    t.EndlessMission(
        begin=("edge-south-SN", 0, 20),
        via=(
            Via("edge-south-SN", lane_offset=40, lane_index=0, required_speed=8,),
            Via("edge-south-SN", lane_offset=60, lane_index=1, required_speed=8,),
            Via("edge-west-EW", lane_offset=10, lane_index=0, required_speed=2,),
            Via("edge-west-EW", lane_offset=20, lane_index=0, required_speed=8,),
            Via("edge-west-EW", lane_offset=50, lane_index=1, required_speed=2,),
            Via("edge-west-EW", lane_offset=55, lane_index=0, required_speed=5,),
            Via("edge-west-EW", lane_offset=60, lane_index=1, required_speed=2,),
            Via("edge-west-EW", lane_offset=65, lane_index=0, required_speed=2,),
            Via("edge-west-EW", lane_offset=70, lane_index=1, required_speed=2,),
        ),
    )
]

social_agent_missions = {
    "all": (
        [
            t.SocialAgentActor(
                name="open-agent", agent_locator="open_agent:open_agent-v0"
            ),
            t.SocialAgentActor(name="rl-agent", agent_locator="rl_agent:rl-agent-v0"),
        ],
        [
            t.Mission(
                t.Route(begin=("edge-west-WE", 1, 10), end=("edge-east-WE", 1, "max"))
            )
        ],
    ),
}

gen_scenario(
    scenario=t.Scenario(
        traffic={"basic": traffic},
        bubbles=bubbles,
        ego_missions=ego_missions,
        social_agent_missions=social_agent_missions,
    ),
    output_dir=Path(__file__).parent,
)
