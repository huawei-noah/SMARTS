from smarts.zoo.registry import make, register
from smarts.core.agent import Agent
from smarts.core.agent_interface import AgentInterface
from smarts.core.controllers import ActionSpaceType
from smarts.zoo import AgentSpec
from smarts.core.observations import Observation


class KeepPoseAgent(Agent):
    def act(self, obs: Observation, **configs):
        return [*obs.ego_vehicle_state.position[:2], obs.ego_vehicle_state.heading, 0.1]


def kp_entrypoint(*args, **kwargs):
    return AgentSpec(
        interface=AgentInterface(debug=True, action=ActionSpaceType.TargetPose),
        agent_builder=KeepPoseAgent,
        agent_params=kwargs,
    )


class MoveToTargetPoseAgent(Agent):
    def __init__(self, target_pose) -> None:
        self._target_pose = target_pose

    def act(self, obs: Observation, **configs):
        return [*self._target_pose.position[:2], self._target_pose.heading, 0.1]


def mtp_entrypoint(target_pose):
    return AgentSpec(
        interface=AgentInterface(debug=True, action=ActionSpaceType.TargetPose),
        agent_builder=MoveToTargetPoseAgent,
        agent_params=[target_pose],
    )


register("keep-pose-v0", kp_entrypoint)
register("move-to-target-pose-v0", mtp_entrypoint)
