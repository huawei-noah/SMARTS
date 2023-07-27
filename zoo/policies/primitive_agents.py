import importlib
import random
from typing import Optional

from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.observations import Observation
from smarts.zoo import Agent, AgentSpec


class RandomLanerAgent(Agent):
    def act(self, obs):
        val = ["keep_lane", "slow_down", "change_lane_left", "change_lane_right"]
        return random.choice(val)


def rla_entrypoint(*, max_episode_steps: Optional[int]) -> AgentSpec:
    return AgentSpec(
        interface=AgentInterface.from_type(
            AgentType.Laner, max_episode_steps=max_episode_steps
        ),
        agent_builder=RandomLanerAgent,
    )


class ChaseViaPointsAgent(Agent):
    def act(self, obs: Observation):
        if (
            len(obs.via_data.near_via_points) < 1
            or obs.ego_vehicle_state.road_id != obs.via_data.near_via_points[0].road_id
        ):
            return (obs.waypoint_paths[0][0].speed_limit, 0)

        nearest = obs.via_data.near_via_points[0]
        if nearest.lane_index == obs.ego_vehicle_state.lane_index:
            return (nearest.required_speed, 0)

        return (
            nearest.required_speed,
            1 if nearest.lane_index > obs.ego_vehicle_state.lane_index else -1,
        )


def cvpa_entrypoint(*, max_episode_steps: Optional[int]):
    return AgentSpec(
        interface=AgentInterface.from_type(
            AgentType.LanerWithSpeed,
            max_episode_steps=max_episode_steps,
        ),
        agent_builder=ChaseViaPointsAgent,
        agent_params=None,
    )


class TrackingAgent(Agent):
    def act(self, obs):
        lane_index = 0
        num_trajectory_points = min([10, len(obs.waypoint_paths[lane_index])])
        # Desired speed is in m/s
        desired_speed = 50 / 3.6
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


def trajectory_tracking_entrypoint(*, max_episode_steps: Optional[int]):
    return AgentSpec(
        interface=AgentInterface.from_type(
            AgentType.Tracker, max_episode_steps=max_episode_steps
        ),
        agent_builder=TrackingAgent,
    )


class StandardLaneFollowerAgent(Agent):
    def act(self, obs):
        return (obs["waypoint_paths"]["speed_limit"][0][0], 0)


def standard_lane_follower_entrypoint(*, max_episode_steps: Optional[int]):
    return AgentSpec(
        interface=AgentInterface.from_type(
            AgentType.LanerWithSpeed, max_episode_steps=max_episode_steps
        ),
        agent_builder=StandardLaneFollowerAgent,
    )
