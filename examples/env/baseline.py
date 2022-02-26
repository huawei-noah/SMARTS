import gym

from smarts.core.agent import AgentSpec
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.env.wrappers.single_agent import SingleAgent

from pathlib import Path

agent_spec = AgentSpec(
    interface=AgentInterface.from_type(
        AgentType.Laner, max_episode_steps=150, rgb=True
    ),
    agent_builder=None,
)


def entry_point(*args, **kwargs):
    from smarts.env.hiway_env import HiWayEnv

    scenario = (Path(__file__).parent / "../../scenarios/figure_eight").resolve()
    ## Note: can build the scenario here
    from cli.studio import _build_single_scenario
    _build_single_scenario(
        clean=True, allow_offset_map=True, scenario=scenario
    )
    return SingleAgent(
        HiWayEnv(
            agent_specs={"agent-007": agent_spec},
            scenarios=[scenario],
            headless=True,
            sumo_headless=True,
        )
    )


gym.register("figure_eight-v0", entry_point=entry_point)
