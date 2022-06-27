from pathlib import Path

import gym

from smarts.zoo.agent_spec import AgentSpec
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.env.wrappers.single_agent import SingleAgent

agent_spec = AgentSpec(
    interface=AgentInterface.from_type(
        AgentType.Laner,
        max_episode_steps=150,
        rgb=True,
        ogm=True,
        drivable_area_grid_map=True,
    ),
    agent_builder=None,
)


def entry_point(*args, **kwargs):
    from smarts.env.hiway_env import HiWayEnv

    scenario = str((Path(__file__).parent / "../../scenarios/figure_eight").resolve())
    ## Note: can build the scenario here
    from smarts.sstudio.build_scenario import build_single_scenario

    build_single_scenario(clean=True, allow_offset_map=True, scenario=scenario)
    hiwayenv = HiWayEnv(
        agent_specs={"agent-007": agent_spec},
        scenarios=[scenario],
        headless=True,
        sumo_headless=True,
    )
    hiwayenv.metadata["render.modes"] = set(hiwayenv.metadata["render.modes"]) | {
        "rgb_array"
    }
    return SingleAgent(
        hiwayenv
    )


gym.register("figure_eight-v0", entry_point=entry_point)
