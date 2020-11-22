from smarts.core.agent import Agent, AgentSpec
from smarts.core.agent_interface import AgentInterface
from smarts.core.controllers import ActionSpaceType
from smarts.zoo.registry import register


class PoseBoidAgent(Agent):
    def act(self, obs):
        returning = {
            vehicle_id: [(idx + 1) * 0.2, 0, (idx + 1) * 0.2]
            for idx, vehicle_id in enumerate(obs)
        }
        return returning


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
        interface=AgentInterface(action=ActionSpaceType.Continuous, waypoints=True),
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
