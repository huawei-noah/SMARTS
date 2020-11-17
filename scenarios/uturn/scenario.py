import os

from smarts.sstudio import gen_traffic, gen_social_agent_missions
from smarts.sstudio import types as t

scenario = os.path.dirname(os.path.realpath(__file__))

open_agent_actor = t.SocialAgentActor(
    name="open-agent", agent_locator="open_agent:open_agent-v0"
)

gen_social_agent_missions(
    scenario,
    [],
    [
        t.Mission(t.Route(begin=("edge-west-WE", 0, 30), end=("edge-south-NS", 0, 40))),
        # Mission(Route(begin=("edge-south-SN", 0, 30), end=("edge-north-SN", 0, 40))),
    ],
)

gen_traffic(
    scenario,
    t.Traffic(
        flows=[
            t.Flow(
                route=t.RandomRoute(), rate=3600, actors={t.TrafficActor(name="car"): 1.0},
            )
        ]
    ),
    name="random",
)

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