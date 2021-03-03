import os

from smarts.sstudio import gen_social_agent_missions
from smarts.sstudio.types import Mission, Route, SocialAgentActor

scenario = os.path.dirname(os.path.realpath(__file__))

laner_agent = SocialAgentActor(
    name="laner-agent",
    agent_locator="scenarios.intersections.2lane_circle.agent_prefabs:laner-agent-v0",
)
buddha_agent = SocialAgentActor(
    name="buddha-agent",
    agent_locator="scenarios.intersections.2lane_circle.agent_prefabs:buddha-agent-v0",
)

gen_social_agent_missions(
    scenario,
    social_agent_actor=laner_agent,
    name=f"s-agent-{laner_agent.name}",
    missions=[Mission(Route(begin=("edge-east-EW", 0, 5), end=("edge-west-EW", 0, 5)))],
)

gen_social_agent_missions(
    scenario,
    social_agent_actor=buddha_agent,
    name=f"s-agent-{buddha_agent.name}",
    missions=[Mission(Route(begin=("edge-west-WE", 0, 5), end=("edge-east-WE", 0, 5)))],
)
