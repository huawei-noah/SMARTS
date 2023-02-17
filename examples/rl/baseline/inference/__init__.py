from smarts.core.agent_interface import AgentInterface, RGB
from smarts.core.controllers import ActionSpaceType
from smarts.zoo.agent_spec import AgentSpec
from smarts.zoo.registry import register

from proximal_policy_optimization import policy

def entry_point(**kwargs):
    return AgentSpec(
        interface=AgentInterface(
            action=ActionSpaceType.ActuatorDynamic,
            top_down_rgb=RGB(
                width = 112,
                height = 112,
                resolution = 50 / 112,
            )
        ),
        agent_builder=policy.Policy,
    )

register(locator="proximal-policy-optimization-agent-v0", entry_point=entry_point)
