import numpy as np

from smarts.core.agent import Agent, AgentSpec
from smarts.core.agent_interface import AgentInterface
from smarts.core.controllers import ActionSpaceType
from smarts.zoo.registry import register


class PoseBoidAgent(Agent):
    def act(self, obs):
        return {vehicle_id: self._single_act(obs_) for vehicle_id, obs_ in obs.items()}

    def _single_act(self, obs):
        wp = obs.waypoint_paths[0][:5][-1]
        dist_to_wp = np.linalg.norm(wp.pos - obs.ego_vehicle_state.position[:2])
        target_speed = 5  # m/s
        return np.array([*wp.pos, wp.heading, dist_to_wp / target_speed])


class TrajectoryBoidAgent(Agent):
    def act(self, obs):
        return {vehicle_id: self._single_act(obs_) for vehicle_id, obs_ in obs.items()}

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
        interface=AgentInterface(action=ActionSpaceType.TargetPose, waypoints=True),
        agent_builder=PoseBoidAgent,
    ),
)

register(
    locator="trajectory-boid-agent-v0",
    entry_point=lambda **kwargs: AgentSpec(
        interface=AgentInterface(action=ActionSpaceType.Trajectory, waypoints=True),
        agent_builder=TrajectoryBoidAgent,
    ),
)
