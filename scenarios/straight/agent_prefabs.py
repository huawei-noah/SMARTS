import numpy as np

from smarts.zoo.registry import register
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.agent import AgentPolicy, AgentSpec
from smarts.core.controllers import ActionSpaceType


class BoidPolicy(AgentPolicy):
    def act(self, obs):
        returning = {
            vehicle_id: self._single_act(obs_) for vehicle_id, obs_ in obs.items()
        }
        return returning

    def _single_act(self, obs):
        wp = obs.waypoint_paths[0][:5][-1]
        dist_to_wp = np.linalg.norm(wp.pos - obs.ego_vehicle_state.position[:2])
        target_speed = 5  # m/s
        return np.array([*wp.pos, wp.heading, dist_to_wp / target_speed])


register(
    locator="boid-agent-v0",
    entry_point=lambda **kwargs: AgentSpec(
        interface=AgentInterface(
            waypoints=True, action=ActionSpaceType.MultiTargetPose
        ),
        policy_builder=BoidPolicy,
    ),
)
