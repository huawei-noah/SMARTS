from smarts.core.agent_interface import ActionSpaceType, AgentInterface
from smarts.zoo.agent_spec import AgentSpec
from zoo.policies.non_interactive_agent import NonInteractiveAgent

AGENT_ID = "Agent-007"

agent_spec = AgentSpec(
    interface=AgentInterface(waypoint_paths=True, action=ActionSpaceType.TargetPose),
    agent_builder=NonInteractiveAgent,
    agent_params={"target_lane_index": {":J3_33": 1, "E3l-3": 1, "E3-35": 1}},
)

agent_specs = {AGENT_ID: agent_spec}
