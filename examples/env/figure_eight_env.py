from pathlib import Path

import gymnasium as gym

from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.env.gymnasium.wrappers.single_agent import SingleAgent
from smarts.env.utils.action_conversion import ActionOptions
from smarts.env.utils.observation_conversion import ObservationOptions

agent_interface = AgentInterface.from_type(
    AgentType.Laner,
    max_episode_steps=150,
    top_down_rgb=True,
    occupancy_grid_map=True,
    drivable_area_grid_map=True,
)


def entry_point(*args, **kwargs):
    scenario = str(
        (Path(__file__).parent / "../../scenarios/sumo/figure_eight").resolve()
    )
    # Note: can build the scenario here
    from smarts.sstudio.scenario_construction import build_scenario

    build_scenario(scenario=scenario, clean=True)
    env = gym.make(
        "smarts.env:hiway-v1",
        agent_interfaces={"agent-007": agent_interface},
        scenarios=[scenario],
        headless=True,
        action_options=ActionOptions.unformatted,
        observation_options=ObservationOptions.unformatted,
        disable_env_checker=True,
    )
    env.metadata["render.modes"] = set(env.metadata.get("render.modes", ()))| {"rgb_array"}
    return SingleAgent(env)


gym.register("figure_eight-v0", entry_point=entry_point)
