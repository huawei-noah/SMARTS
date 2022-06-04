from pathlib import Path

from smarts.sstudio import gen_scenario
from smarts.sstudio import types as t

trajectory_boid_agent = t.BoidAgentActor(
    name="trajectory-boid",
    agent_locator="scenarios.straight.agent_prefabs:trajectory-boid-agent-v0",
)

pose_boid_agent = t.BoidAgentActor(
    name="pose-boid",
    agent_locator="scenarios.straight.agent_prefabs:pose-boid-agent-v0",
)

traffic = t.Traffic(
    flows=[
        t.Flow(
            route=t.Route(
                begin=("west", lane_idx, 0),
                end=("east", lane_idx, "max"),
            ),
            rate=50,
            actors={t.TrafficActor("car"): 1},
        )
        for lane_idx in range(3)
    ]
)

scenario = t.Scenario(
    traffic={"all": traffic},
    bubbles=[
        t.Bubble(
            zone=t.PositionalZone(pos=(50, 0), size=(40, 20)),
            margin=5,
            actor=trajectory_boid_agent,
        ),
        t.Bubble(
            zone=t.PositionalZone(pos=(150, 0), size=(50, 20)),
            margin=5,
            actor=pose_boid_agent,
            keep_alive=True,
        ),
    ],
)

gen_scenario(scenario, output_dir=str(Path(__file__).parent))
