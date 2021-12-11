import os
import pickle

from smarts.sstudio import gen_social_agents
from smarts.sstudio import types as t

scenario = os.path.dirname(os.path.realpath(__file__))

with open(os.environ["SOCIAL_AGENT_PATH"], "rb") as f:
    social_agent = pickle.load(f)

gen_social_agents(
    scenario,
    name=f"s-agent-{social_agent.name}",
    social_actor_mission_pairs=[
        (
            social_agent,
            t.Mission(t.Route(begin=("E3l-3", 1, 300), end=("E3-35", 1, "max"))),
        )
    ],
)
