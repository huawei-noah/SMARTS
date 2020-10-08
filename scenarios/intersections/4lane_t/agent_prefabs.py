import numpy as np

from smarts.core.agent import AgentPolicy, AgentSpec
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.controllers import ActionSpaceType
from smarts.zoo.registry import register


class KeepLanePolicy(AgentPolicy):
    def act(self, obs):
        return "keep_lane"


class MotionPlannerPolicy(AgentPolicy):
    def act(self, obs):
        wp = obs.waypoint_paths[0][:5][-1]
        dist_to_wp = np.linalg.norm(wp.pos - obs.ego_vehicle_state.position[:2])
        target_speed = 5  # m/s
        return np.array([*wp.pos, wp.heading, dist_to_wp / target_speed])


register(
    locator="zoo-agent-v0",
    entry_point=lambda **kwargs: AgentSpec(
        interface=AgentInterface.from_type(AgentType.Laner, max_episode_steps=20000),
        policy_builder=KeepLanePolicy,
    ),
)

register(
    locator="motion-planner-agent-v0",
    entry_point=lambda **kwargs: AgentSpec(
        interface=AgentInterface(waypoints=True, action=ActionSpaceType.TargetPose),
        policy_builder=MotionPlannerPolicy,
    ),
)
