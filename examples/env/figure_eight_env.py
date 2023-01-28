from pathlib import Path

import gym

from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.env.wrappers.single_agent import SingleAgent
from smarts.zoo.agent_spec import AgentSpec

agent_spec = AgentSpec(
    interface=AgentInterface.from_type(
        AgentType.Laner,
        max_episode_steps=150,
        top_down_rgb=True,
        occupancy_grid_map=True,
        drivable_area_grid_map=True,
    ),
    agent_builder=None,
)


def entry_point(*args, **kwargs):
    from smarts.env.hiway_env import HiWayEnv

    scenario = str(
        (Path(__file__).parent / "../../scenarios/sumo/figure_eight").resolve()
    )
    ## Note: can build the scenario here
    from smarts.sstudio.scenario_construction import build_scenario

    build_scenario(scenario=scenario, clean=True)
    hiwayenv = HiWayEnv(
        agent_specs={"agent-007": agent_spec},
        scenarios=[scenario],
        headless=True,
        sumo_headless=True,
    )
    hiwayenv.metadata["render.modes"] = set(hiwayenv.metadata["render.modes"]) | {
        "rgb_array"
    }
    return SingleAgent(hiwayenv)


gym.register("figure_eight-v0", entry_point=entry_point)
