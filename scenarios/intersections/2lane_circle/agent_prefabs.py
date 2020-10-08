from smarts.zoo.registry import register
from smarts.core.agent_interface import AgentInterface, DoneCriteria
from smarts.core.agent import AgentPolicy, AgentSpec
from smarts.core.controllers import ActionSpaceType


class KeepLanePolicy(AgentPolicy):
    def act(self, obs):
        return "keep_lane"


class StoppedPolicy(AgentPolicy):
    def act(self, obs):
        return (0, 0)


register(
    locator="laner-agent-v0",
    entry_point=lambda **kwargs: AgentSpec(
        interface=AgentInterface(
            waypoints=True, action=ActionSpaceType.Lane, max_episode_steps=5000
        ),
        policy_builder=KeepLanePolicy,
    ),
)

register(
    locator="buddha-agent-v0",
    entry_point=lambda **kwargs: AgentSpec(
        interface=AgentInterface(
            waypoints=True,
            action=ActionSpaceType.LaneWithContinuousSpeed,
            done_criteria=DoneCriteria(not_moving=True),
        ),
        policy_builder=StoppedPolicy,
    ),
)
