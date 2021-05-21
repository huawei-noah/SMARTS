# MIT License
#
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
import collections
import dataclasses
import enum
from smarts.core.controllers import ActionSpaceType
from smarts.core.coordinates import Heading
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple, Union
import numpy as np
import torch
from scipy.spatial import distance
import random, math, gym
from sys import path
from collections import OrderedDict
from smarts.core.agent_interface import (
    Accelerometer,
    DrivableAreaGridMap,
    Lidar,
    NeighborhoodVehicles,
    OGM,
    RGB,
    RoadWaypoints,
    Waypoints
)
from smarts.core.sensors import Observation, VehicleObservation

path.append("./ultra")
from ultra.utils.common import (
    get_closest_waypoint,
    get_path_to_goal,
    ego_social_safety,
    rotate2d_vector,
)

seed = 0
random.seed(seed)
num_lookahead = 100


class ActionType(enum.Enum):
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"


class ObservationType(enum.Enum):
    VECTOR = "vector"
    IMAGE = "image"


class RewardType(enum.Enum):
    DEFAULT = "default"


# TODO: Should create separate modules for each adapter since they are all purely static
#       classes. E.g. adapters/__init__.py that contains the functions at the bottom of
#       this file, and adapters/default_adapters.py that contains the default adapters.


class Adapter:
    gym_space: gym.Space
    required_interface: Dict[str, Any]

    @staticmethod
    def adapt(*args, **kwargs):
        raise NotImplementedError


class DiscreteActionAdapter(Adapter):
    gym_space: gym.Space = None  # TODO: Make gym space.
    required_interface = { "action": ActionSpaceType.Lane }

    @staticmethod
    def adapt(action):
        return action


class ContinuousActionAdapter(Adapter):
    gym_space: gym.Space = gym.spaces.Box(
        low=np.array([0.0, 0.0, -1.0]),
        high=np.array([1.0, 1.0, 1.0]),
        dtype=np.float32,
    )
    required_interface = { "action": ActionSpaceType.Continuous }

    @staticmethod
    def adapt(action):
        return action


class VectorObservationAdapter(Adapter):
    _WAYPOINTS = 20  # Number of waypoints on the path ahead of the ego vehicle.
    _SIZE = (
        1 +               # Speed.
        1 +               # Distance from center.
        1 +               # Steering.
        1 +               # Angle error.
        2 +               # Relative goal position.
        2 * _WAYPOINTS +  # Waypoints lookahead.
        1                 # Road speed.
    )
    _NORMALIZATION_VALUES = {
        "speed": 30.0,
        "distance_from_center": 1.0,
        "steering": 3.14,  # In radians.
        "angle_error": 3.14,  # In radians.
        "relative_goal_position": 100.0,
        "waypoints_lookahead": 10.0,
        "road_speed": 30.0,
    }
    _RADIUS = 200.0  # Locate all social vehicles within this radius of the ego vehicle.
    _CAPACITY = 10  # Number of social vehicles we keep in the adapted observation.
    _FEATURES = 4  # Number of features for each social vehicle.

    gym_space: gym.Space = gym.spaces.Dict(
        {
            "low_dim_states": gym.spaces.Box(
                low=-1e10, high=1e10, shape=(_SIZE,), dtype=np.float32,
            ),
            "social_vehicles": gym.spaces.Box(
                low=-1e10, high=1e10, shape=(_CAPACITY, _FEATURES), dtype=np.float32,
            ),
        }
    )
    required_interface = {
        "waypoints": Waypoints(lookahead=_WAYPOINTS),
        "neighborhood_vehicles": NeighborhoodVehicles(radius=_RADIUS),
    }

    @staticmethod
    def adapt(observation: Observation):
        ego_position = observation.ego_vehicle_state.position
        ego_heading = observation.ego_vehicle_state.heading
        ego_speed = observation.ego_vehicle_state.speed
        ego_steering = observation.ego_vehicle_state.steering
        ego_start = observation.ego_vehicle_state.mission.start
        ego_goal = observation.ego_vehicle_state.mission.goal
        ego_waypoints = observation.waypoint_paths

        ego_goal_path = get_path_to_goal(
            goal=ego_goal,
            paths=ego_waypoints,
            start=ego_start
        )
        ego_closest_waypoint, ego_lookahead_waypoints = get_closest_waypoint(
            num_lookahead=VectorObservationAdapter._WAYPOINTS,
            goal_path=ego_goal_path,
            ego_position=ego_position,
            ego_heading=ego_heading
        )
        ego_lookahead_waypoints = np.hstack(ego_lookahead_waypoints)

        signed_distance_from_center = ego_closest_waypoint.signed_lateral_error(
            ego_position
        )
        lane_width = ego_closest_waypoint.lane_width * 0.5
        ego_distance_from_center = signed_distance_from_center / lane_width

        ego_relative_rotated_goal_position = rotate2d_vector(
            np.asarray(ego_goal.position[0:2]) - np.asarray(ego_position[0:2]),
            -ego_heading
        )

        observation_dict = dict(
            speed=ego_speed,
            relative_goal_position=ego_relative_rotated_goal_position,
            distance_from_center=ego_distance_from_center,
            steering=ego_steering,
            angle_error=ego_closest_waypoint.relative_heading(ego_heading),
            road_speed=ego_closest_waypoint.speed_limit,
            start=ego_start.position,
            goal=ego_goal.position,
            heading=ego_heading,
            goal_path=ego_goal_path,
            ego_position=ego_position,
            waypoint_paths=ego_waypoints,
            events=observation.events,
            waypoints_lookahead=ego_lookahead_waypoints
        )
        normalized_observation = [
            VectorObservationAdapter._normalize(key, observation_dict[key])
            for key in VectorObservationAdapter._NORMALIZATION_VALUES.keys()
        ]
        low_dim_states_observation = np.concatenate([
            value
            if isinstance(value, collections.abc.Iterable)
            else np.asarray([value]).astype(np.float32)
            for value in normalized_observation
        ], axis=-1)

        # Adapt the social vehicles.
        social_vehicles = observation.neighborhood_vehicle_states

        if len(social_vehicles) == 0:
            # There are no social vehicles. Create an empty array with the correct
            # number of features so it can be padded.
            social_vehicles = np.empty(
                (0, VectorObservationAdapter._FEATURES), dtype=np.float32
            )
        else:
            # Sort by distance to the ego vehicle.
            social_vehicles.sort(
                key=lambda vehicle: VectorObservationAdapter._get_distance(vehicle, ego_position), reverse=False
            )
            # Extract the state of each social vehicle.
            social_vehicles = np.asarray([
                VectorObservationAdapter._extract_social_vehicle_state(
                    social_vehicle=social_vehicle,
                    ego_position=ego_position,
                    ego_heading=ego_heading,
                )
                for social_vehicle in social_vehicles
            ], dtype=np.float32)

        # Pad with zero vectors if we don't have enough social vehicles.
        if len(social_vehicles) < VectorObservationAdapter._CAPACITY:
            remain = VectorObservationAdapter._CAPACITY - len(social_vehicles)
            empty_social_vehicles = np.zeros(
                shape=(remain, VectorObservationAdapter._FEATURES), dtype=np.float32
            )
            social_vehicles = np.concatenate((social_vehicles, empty_social_vehicles))

        # Remove extra social vehicles if there were too many in the observation.
        social_vehicles = social_vehicles[:VectorObservationAdapter._CAPACITY]

        vector_observation = {
            "low_dim_states": low_dim_states_observation,
            "social_vehicles": social_vehicles,
        }
        return vector_observation

    def _normalize(key, value):
        if key not in VectorObservationAdapter._NORMALIZATION_VALUES:
            return value
        return value / VectorObservationAdapter._NORMALIZATION_VALUES[key]

    def _extract_social_vehicle_state(
        social_vehicle: VehicleObservation,
        ego_position: Tuple[float, float, float],
        ego_heading: Heading,
    ) -> List[float]:
        absolute_position_difference = social_vehicle.position[:2] - ego_position[:2]
        relative_position_difference = rotate2d_vector(
            absolute_position_difference, -ego_heading
        )
        absolute_heading_difference = social_vehicle.heading - ego_heading

        # NOTE: The number of elements in this list should equal _FEATURES.
        social_vehicle_state = [
            relative_position_difference[0] / 100.0,
            relative_position_difference[1] / 100.0,
            absolute_heading_difference / 3.14,
            social_vehicle.speed / 30.0
        ]
        return social_vehicle_state

    def _get_distance(
        social_vehicle: VehicleObservation, ego_position: Tuple[float, float, float]
    ) -> float:
        return distance.euclidean(social_vehicle.position[:2], ego_position[:2])


class ImageObservationAdapter(Adapter):
    _WIDTH = 64
    _HEIGHT = 64
    _RESOLUTION = 50 / 64

    gym_space: gym.Space = gym.spaces.Box(
        low=-1.0, high=1.0, shape=(_HEIGHT, _WIDTH), dtype=np.float32
    )
    required_interface={
        "rgb": RGB(width=_WIDTH, height=_HEIGHT, resolution=_RESOLUTION)
    }

    @staticmethod
    def adapt(observation: Observation):
        raise NotImplementedError  # TODO: Fill in.


class DefaultRewardAdapter(Adapter):
    _WAYPOINTS = 20
    _RADIUS = 200.0

    # TODO: Reward doesn't need a gym space.
    gym_space: gym.Space = gym.spaces.Box(
        low=-1e10, high=1e10, shape=(1,), dtype=np.float32
    )
    required_interface={
        "waypoints": Waypoints(lookahead=_WAYPOINTS),
        "neighborhood_vehicles": NeighborhoodVehicles(radius=_RADIUS),
    }

    @staticmethod
    def adapt(observation: Observation, reward: float):
        env_reward = reward
        ego_events = observation.events
        ego_observation = observation.ego_vehicle_state
        start = observation.ego_vehicle_state.mission.start
        goal = observation.ego_vehicle_state.mission.goal
        path = get_path_to_goal(
            goal=goal, paths=observation.waypoint_paths, start=start
        )

        linear_jerk = np.linalg.norm(ego_observation.linear_jerk)
        angular_jerk = np.linalg.norm(ego_observation.angular_jerk)

        # Distance to goal
        ego_2d_position = ego_observation.position[0:2]
        goal_dist = distance.euclidean(ego_2d_position, goal.position)

        closest_wp, _ = get_closest_waypoint(
            num_lookahead=num_lookahead,
            goal_path=path,
            ego_position=ego_observation.position,
            ego_heading=ego_observation.heading,
        )
        angle_error = closest_wp.relative_heading(
            ego_observation.heading
        )  # relative heading radians [-pi, pi]

        # Distance from center
        signed_dist_from_center = closest_wp.signed_lateral_error(
            observation.ego_vehicle_state.position
        )
        lane_width = closest_wp.lane_width * 0.5
        ego_dist_center = signed_dist_from_center / lane_width

        # number of violations
        (ego_num_violations, social_num_violations,) = ego_social_safety(
            observation,
            d_min_ego=1.0,
            t_c_ego=1.0,
            d_min_social=1.0,
            t_c_social=1.0,
            ignore_vehicle_behind=True,
        )

        speed_fraction = max(0, ego_observation.speed / closest_wp.speed_limit)
        ego_step_reward = 0.02 * min(speed_fraction, 1) * np.cos(angle_error)
        ego_speed_reward = min(
            0, (closest_wp.speed_limit - ego_observation.speed) * 0.01
        )  # m/s
        ego_collision = len(ego_events.collisions) > 0
        ego_collision_reward = -1.0 if ego_collision else 0.0
        ego_off_road_reward = -1.0 if ego_events.off_road else 0.0
        ego_off_route_reward = -1.0 if ego_events.off_route else 0.0
        ego_wrong_way = -0.02 if ego_events.wrong_way else 0.0
        ego_goal_reward = 0.0
        ego_time_out = 0.0
        ego_dist_center_reward = -0.002 * min(1, abs(ego_dist_center))
        ego_angle_error_reward = -0.005 * max(0, np.cos(angle_error))
        ego_reached_goal = 1.0 if ego_events.reached_goal else 0.0
        ego_safety_reward = -0.02 if ego_num_violations > 0 else 0
        social_safety_reward = -0.02 if social_num_violations > 0 else 0
        ego_lat_speed = 0.0  # -0.1 * abs(long_lat_speed[1])
        ego_linear_jerk = -0.0001 * linear_jerk
        ego_angular_jerk = -0.0001 * angular_jerk * math.cos(angle_error)
        env_reward /= 100
        # DG: Different speed reward
        ego_speed_reward = -0.1 if speed_fraction >= 1 else 0.0
        ego_speed_reward += -0.01 if speed_fraction < 0.01 else 0.0

        rewards = sum(
            [
                ego_goal_reward,
                ego_collision_reward,
                ego_off_road_reward,
                ego_off_route_reward,
                ego_wrong_way,
                ego_speed_reward,
                # ego_time_out,
                ego_dist_center_reward,
                ego_angle_error_reward,
                ego_reached_goal,
                ego_step_reward,
                env_reward,
                # ego_linear_jerk,
                # ego_angular_jerk,
                # ego_lat_speed,
                # ego_safety_reward,
                # social_safety_reward,
            ]
        )
        return rewards


# NOTE: Another idea for how the baseline adapter could work???
#
# import ultra.baselines.adapter as adapter
#
# action_type = params.action_type
# observation_type = params.observation_type
# reward_type = params.reward_type
#
# required_interface = adapter.required_interface_from_types(
#     action_type=action_type,
#     observation_type=observation_type,
#     reward_type=reward_type,
# )
#
# spec = AgentSpec(
#     interface=AgentInterface(
#         **required_interface,
#         max_episode_steps=max_episode_steps,
#         debug=True,
#     ),
#     agent_params=dict(
#         policy_params=params, checkpoint_dir=checkpoint_dir
#     ),
#     agent_builder=policy_class,
#     action_adapter=adapter.action_adapter_from_type(action_type)
#     observation_adapter=adapter.observation_adapter_from_type(observation_type),
#     reward_adapter=adapter.reward_adapter_from_type(reward_type),
# )
#

# TODO: Have the keys be enums.
_TYPE_TO_ADAPTER: Dict[str, Adapter] = {
    "continuous": ContinuousActionAdapter,
    "discrete": DiscreteActionAdapter,
    "vector": VectorObservationAdapter,
    "image": ImageObservationAdapter,
    "default": DefaultRewardAdapter,
}


def action_space_from_type(action_type: ActionType) -> gym.Space:
    return _TYPE_TO_ADAPTER[action_type].gym_space


def action_adapter_from_type(action_type: ActionType) -> Callable:
    return _TYPE_TO_ADAPTER[action_type].adapt


def observation_space_from_type(
    observation_type: ObservationType
) -> gym.Space:
    return _TYPE_TO_ADAPTER[observation_type].gym_space


def observation_adapter_from_type(
    observation_type: ObservationType
) -> Callable:
    return _TYPE_TO_ADAPTER[observation_type].adapt


def reward_adapter_from_type(reward_type: RewardType) -> Callable:
    return _TYPE_TO_ADAPTER[reward_type].adapt


def required_interface_from_types(
    action_type: ActionType,
    observation_type: ObservationType,
    reward_type: RewardType,
) -> Dict[str, Any]:
    # TODO: If we are using the Types we have to convert them from strings here first.

    required_interface = {}

    # TODO: Make this nicer.

    action_interface = _TYPE_TO_ADAPTER[action_type].required_interface
    observation_interface = _TYPE_TO_ADAPTER[observation_type].required_interface
    reward_interface = _TYPE_TO_ADAPTER[reward_type].required_interface

    for interface_name, interface in action_interface.items():
        if interface_name in required_interface:
            # TODO: Does this actually compare the interfaces correctly?
            assert required_interface[interface_name] == interface
        else:
            required_interface[interface_name] = interface

    for interface_name, interface in observation_interface.items():
        if interface_name in required_interface:
            # TODO: Does this actually compare the interfaces correctly?
            assert required_interface[interface_name] == interface
        else:
            required_interface[interface_name] = interface

    for interface_name, interface in reward_interface.items():
        if interface_name in required_interface:
            # TODO: Does this actually compare the interfaces correctly?
            assert required_interface[interface_name] == interface
        else:
            required_interface[interface_name] = interface

    return required_interface
