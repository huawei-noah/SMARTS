from smarts.core.agent import AgentSpec
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.controllers import ActionSpaceType
from smarts.zoo.registry import register

from .keep_lane_policy import KeepLanePolicy
from .non_interactive_policy import NonInteractivePolicy


register(
    locator="non-interactive-agent-v0",
    entry_point=lambda **kwargs: AgentSpec(
        interface=AgentInterface(waypoints=True, action=ActionSpaceType.TargetPose),
        policy_builder=NonInteractivePolicy,
        policy_params=kwargs,
    ),
)

register(
    locator="keep-lane-agent-v0",
    entry_point=lambda **kwargs: AgentSpec(
        interface=AgentInterface.from_type(AgentType.Laner, max_episode_steps=20000),
        policy_builder=KeepLanePolicy,
    ),
)
