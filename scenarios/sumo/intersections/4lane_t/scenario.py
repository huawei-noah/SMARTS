from pathlib import Path

from smarts.core.colors import Colors
from smarts.sstudio import gen_scenario
from smarts.sstudio import types as t

traffic = t.Traffic(
    flows=[
        t.Flow(
            route=t.Route(
                begin=("edge-west-WE", 0, 10), end=("edge-east-WE", 0, "max")
            ),
            repeat_route=True,
            rate=400,
            actors={t.TrafficActor("car"): 1},
        ),
        t.Flow(
            route=t.Route(
                begin=("edge-east-EW", 0, 10), end=("edge-west-EW", 0, "max")
            ),
            repeat_route=True,
            rate=400,
            actors={t.TrafficActor("car"): 1},
        ),
    ]
)

agent_prefabs = "scenarios.sumo.intersections.4lane_t.agent_prefabs"

motion_planner_actor = t.SocialAgentActor(
    name="motion-planner-agent",
    agent_locator=f"{agent_prefabs}:motion-planner-agent-v0",
)

zoo_agent_actor = t.SocialAgentActor(
    name="zoo-agent",
    agent_locator=f"{agent_prefabs}:zoo-agent-v0",
)
# Replace the above lines with the code below if you want to replay the agent actions and inputs
# zoo_agent_actor = t.SocialAgentActor(
#     name="zoo-agent",
#     agent_locator="zoo.policies:replay-agent-v0",
#     policy_kwargs={
#         "save_directory": "./replay",
#         "id": "agent_za",
#         "wrapped_agent_locator": f"{agent_prefabs}:zoo-agent-v0",
#     },
# )


# motion_planner_actor = t.SocialAgentActor(
#     name="motion-planner-agent",
#     agent_locator="zoo.policies:replay-agent-v0",
#     policy_kwargs={
#         "save_directory": "./replay",
#         "id": "agent_mp",
#         "wrapped_agent_locator": f"{agent_prefabs}:motion-planner-agent-v0",
#     },
# )


bubbles = [
    t.Bubble(
        zone=t.MapZone(start=("edge-west-WE", 0, 50), length=10, n_lanes=1),
        margin=2,
        actor=zoo_agent_actor,
    ),
    t.Bubble(
        zone=t.PositionalZone(pos=(100, 100), size=(20, 20)),
        margin=2,
        actor=motion_planner_actor,
    ),
]

ego_missions = [
    t.EndlessMission(
        begin=("edge-south-SN", 1, 20),
    )
]

social_agent_missions = {
    "all": (
        [
            t.SocialAgentActor(
                name="keep-lane-agent-v0",
                agent_locator="zoo.policies:keep-lane-agent-v0",
            ),
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
        scenario_metadata=t.ScenarioMetadata(r".*-1.*", Colors.Yellow),
    ),
    output_dir=Path(__file__).parent,
)
