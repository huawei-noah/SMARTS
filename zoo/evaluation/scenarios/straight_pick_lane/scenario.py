import os
import pickle

from smarts.sstudio.genscenario import _gen_social_agent_missions
from smarts.sstudio import types as t

scenario = os.path.dirname(os.path.realpath(__file__))


with open(os.environ["SOCIAL_AGENT_PATH"], "rb") as f:
    social_agent = pickle.load(f)

_gen_social_agent_missions(
    scenario,
    social_agent_actor=social_agent,
    name=f"s-agent-{social_agent.name}",
    missions=[
        t.Mission(t.Route(begin=("E35-3", 1, 70), end=("E3-3l", 0, 30))),
    ],
)
