from smarts.core.agent_interface import AgentInterface, RGB
from smarts.core.controllers import ActionSpaceType
from smarts.zoo.agent_spec import AgentSpec
from smarts.zoo.registry import register

from contrib_policy.policy import Policy

def entry_point(**kwargs):
    interface = AgentInterface(
        action=ActionSpaceType.ActuatorDynamic,
        top_down_rgb=RGB(
            width = 112,
            height = 112,
            resolution = 50 / 112,
        )
    )
    return AgentSpec(
        interface=interface,
        agent_builder=Policy,
    )

register(locator="contrib-agent-v0", entry_point=entry_point)
