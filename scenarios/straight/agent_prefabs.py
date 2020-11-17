import numpy as np

from smarts.zoo.registry import register
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.agent import AgentPolicy, AgentSpec
from smarts.core.controllers import ActionSpaceType


class PoseBoidPolicy(AgentPolicy):
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


class TrajectoryBoidPolicy(AgentPolicy):
    def act(self, obs):
        returning = {
            vehicle_id: self._single_act(obs_) for vehicle_id, obs_ in obs.items()
        }
        return returning

    def _single_act(self, obs):
        lane_index = 0
        num_trajectory_points = min([10, len(obs.waypoint_paths[lane_index])])
        desired_speed = 50 / 3.6  # m/s
        trajectory = [
            [
                obs.waypoint_paths[lane_index][i].pos[0]
                for i in range(num_trajectory_points)
            ],
            [
                obs.waypoint_paths[lane_index][i].pos[1]
                for i in range(num_trajectory_points)
            ],
            [
                obs.waypoint_paths[lane_index][i].heading
                for i in range(num_trajectory_points)
            ],
            [desired_speed for i in range(num_trajectory_points)],
        ]
        return trajectory


register(
    locator="pose-boid-agent-v0",
    entry_point=lambda **kwargs: AgentSpec(
        interface=AgentInterface(
            action=ActionSpaceType.MultiTargetPose,
            waypoints=True,
            ogm=True,
            rgb=True,
            drivable_area_grid_map=True,
        ),
        policy_builder=PoseBoidPolicy,
    ),
)

register(
    locator="trajectory-boid-agent-v0",
    entry_point=lambda **kwargs: AgentSpec(
        interface=AgentInterface(action=ActionSpaceType.Trajectory, waypoints=True,),
        policy_builder=TrajectoryBoidPolicy,
    ),
)
