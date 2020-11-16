from smarts.core.agent import AgentSpec
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.controllers import ActionSpaceType
from smarts.zoo.registry import register

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
