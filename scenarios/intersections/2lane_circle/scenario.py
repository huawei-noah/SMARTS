import os
from pathlib import Path

from smarts.sstudio import gen_social_agents, gen_scenario
from smarts.sstudio.types import Mission, Route, SocialAgentActor, Scenario

scenario = os.path.dirname(os.path.realpath(__file__))

laner_agent = SocialAgentActor(
    name="laner-agent",
    agent_locator="scenarios.intersections.2lane_circle.agent_prefabs:laner-agent-v0",
)
buddha_agent = SocialAgentActor(
    name="buddha-agent",
    agent_locator="scenarios.intersections.2lane_circle.agent_prefabs:buddha-agent-v0",
)

social_agents = {
    f"all": [
        (
            buddha_agent,
            Mission(Route(begin=("edge-west-WE", 0, 5), end=("edge-east-WE", 0, 5))),
        ),
        (
            laner_agent,
            Mission(Route(begin=("edge-east-EW", 0, 5), end=("edge-west-EW", 0, 5))),
        ),
    ]
}

gen_scenario(
    scenario=Scenario(
        social_agent_groups=social_agents,
    ),
    output_dir=Path(__file__).parent,
)
