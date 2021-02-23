from typing import Any, Dict

from smarts.core.agent import AgentSpec
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.controllers import ActionSpaceType
from smarts.zoo.registry import make, register

from .keep_lane_agent import KeepLaneAgent
from .non_interactive_agent import NonInteractiveAgent

register(
    locator="non-interactive-agent-v0",
    entry_point=lambda **kwargs: AgentSpec(
        interface=AgentInterface(waypoints=True, action=ActionSpaceType.TargetPose),
        agent_builder=NonInteractiveAgent,
        agent_params=kwargs,
    ),
)

register(
    locator="keep-lane-agent-v0",
    entry_point=lambda **kwargs: AgentSpec(
        interface=AgentInterface.from_type(AgentType.Laner, max_episode_steps=20000),
        agent_builder=KeepLaneAgent,
    ),
)

social_index = 0


def replay_entrypoint(
    save_directory,
    id,
    wrapped_agent_locator,
    read=True,
):
    from .replay_agent import ReplayAgent

    internal_spec = make(wrapped_agent_locator)
    global social_index
    spec = AgentSpec(
        interface=internal_spec.interface,
        agent_params={
            "save_directory": save_directory,
            "id": f"{id}_{social_index}",
            "internal_spec": internal_spec,
            "read": read,
        },
        agent_builder=ReplayAgent,
    )
    social_index += 1
    return spec


register(locator="replay-agent-v0", entry_point=replay_entrypoint)
