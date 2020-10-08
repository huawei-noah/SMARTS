from pathlib import Path

from smarts.sstudio import types as t
from smarts.sstudio import gen_scenario

boid_agent = t.BoidAgentActor(
    name="hive-mind", agent_locator="scenarios.straight.agent_prefabs:boid-agent-v0",
)

traffic = t.Traffic(
    flows=[
        t.Flow(
            route=t.Route(begin=("west", lane_idx, 0), end=("east", lane_idx, "max"),),
            rate=50,
            actors={t.TrafficActor("car"): 1,},
        )
        for lane_idx in range(3)
    ]
)

scenario = t.Scenario(
    traffic={"all": traffic},
    bubbles=[
        t.Bubble(
            zone=t.PositionalZone(pos=(100, 0), size=(30, 20)),
            margin=5,
            actor=boid_agent,
        ),
    ],
)

gen_scenario(scenario, output_dir=str(Path(__file__).parent))
