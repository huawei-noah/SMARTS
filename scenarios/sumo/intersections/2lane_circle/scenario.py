from pathlib import Path

import smarts.sstudio.sstypes as t
from smarts.sstudio import gen_scenario

laner_agent = t.SocialAgentActor(
    name="laner-agent",
    agent_locator="scenarios.sumo.intersections.2lane_circle.agent_prefabs:laner-agent-v0",
)
buddha_agent = t.SocialAgentActor(
    name="buddha-agent",
    agent_locator="scenarios.sumo.intersections.2lane_circle.agent_prefabs:buddha-agent-v0",
)

# Replace the above lines with the code below if you want to replay the agent actions and inputs
# laner_agent = t.SocialAgentActor(
#     name="laner-agent",
#     agent_locator="zoo.policies:replay-agent-v0",
#     policy_kwargs={
#         "save_directory": "./replay",
#         "id": "agent_la",
#         "wrapped_agent_locator": "scenarios.sumo.intersections.2lane_circle.agent_prefabs:laner-agent-v0",
#     },
# )


# buddha_agent = t.SocialAgentActor(
#     name="buddha-agent",
#     agent_locator="zoo.policies:replay-agent-v0",
#     policy_kwargs={
#         "save_directory": "./replay",
#         "id": "agent_ba",
#         "wrapped_agent_locator": "scenarios.sumo.intersections.2lane_circle.agent_prefabs:buddha-agent-v0",
#     },
# )

gen_scenario(
    scenario=t.Scenario(
        social_agent_missions={
            f"s-agent-{laner_agent.name}": (
                [laner_agent],
                [
                    t.Mission(
                        t.Route(
                            begin=("edge-east-EW", 0, 5), end=("edge-west-EW", 0, 5)
                        )
                    )
                ],
            ),
            f"s-agent-{buddha_agent.name}": (
                [buddha_agent],
                [
                    t.Mission(
                        t.Route(
                            begin=("edge-west-WE", 0, 5), end=("edge-east-WE", 0, 5)
                        )
                    )
                ],
            ),
        }
    ),
    output_dir=Path(__file__).parent,
)
