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
from ultra.baselines.common.baseline_state_preprocessor import BaselineStatePreprocessor
from ultra.baselines.common.social_vehicle_config import get_social_vehicle_configs
from ultra.baselines.common.yaml_loader import load_yaml

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


class AdapterType(enum.Enum):
    pass


class ActionType(enum.Enum):
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"


class ObservationType(enum.Enum):
    VECTOR = "vector"
    IMAGE = "image"


class RewardType(enum.Enum):
    DEFAULT = "default"


class ObservationAdapter:
    _ACCEPTED_INTERFACES = [
        DrivableAreaGridMap,
        OGM,
        RGB,
        Lidar,
        Waypoints,
        RoadWaypoints,
        NeighborhoodVehicles,
        Accelerometer,
    ]

    def __init__(
        self,
        name: str,
        gym_space: gym.spaces,
        size: int,
        required_interfaces: Dict[str, Any],
    ):
        self._name = name
        self._gym_space = gym_space
        self._size = size
        self._required_interfaces = required_interfaces

        for interface in self._required_interfaces:
            if type(interface) not in self._ACCEPTED_INTERFACES:
                raise f"{type(interface)} is not an accepted observation interface."

    @property
    def name(self) -> str:
        return self._name

    @property
    def gym_space(self) -> gym.spaces:
        return self._gym_space

    @property
    def size(self) -> int:
        return self._size

    @property
    def required_interfaces(
        self,
    ) -> Dict[
        str,
        Union[
            DrivableAreaGridMap,
            OGM,
            RGB,
            Lidar,
            Waypoints,
            RoadWaypoints,
            NeighborhoodVehicles,
            Accelerometer,
        ],
    ]:
        return self._required_interfaces

    # TODO: This could be a static method if we don't keep any instance attributes.
    def adapt(self, observation: Observation):
        raise NotImplementedError


class TopDownRgbObservationAdapter(ObservationAdapter):
    _WIDTH = 64
    _HEIGHT = 64
    _RESOLUTION = 50 / 64

    def __init__(self):
        super(TopDownRgbObservationAdapter, self).__init__(
            name="top_down_rgb",
            gym_space=gym.spaces.Box(low=-1.0, high=1.0, shape=(3, self._HEIGHT, self._WIDTH)),
            size=self._WIDTH * self._HEIGHT,
            required_interfaces={
                "rgb": RGB(width=self._WIDTH, height=self._HEIGHT, resolution=self._RESOLUTION)
            }
        )

    def adapt(self, observation: Observation):
        # TODO: Make into grayscale and support stacking.
        return observation.top_down_rgb


class LowDimStatesObservationAdapter(ObservationAdapter):
    _WAYPOINTS = 20
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
        # "action": 1.0,  # 2
        "waypoints_lookahead": 10.0,
        "road_speed": 30.0,
    }

    def __init__(self):
        super(LowDimStatesObservationAdapter, self).__init__(
            name="low_dim_states",
            gym_space=gym.spaces.Box(
                low=-1e10,
                high=1e10,
                shape=(self._SIZE,),
                dtype=torch.Tensor,
            ),
            size=None,  # TODO: Fill this out.
            required_interfaces={
                "waypoints": Waypoints(lookahead=self._WAYPOINTS)
            }
        )

    def adapt(self, observation: Observation):
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
            num_lookahead=100,
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
        )
        normalized_observation = [
            self._normalize(key, observation_dict[key])
            for key in self._NORMALIZATION_VALUES.keys()
        ]
        low_dim_states_observation = np.concatenate([
            value
            if isinstance(value, collections.abc.Iterable)
            else np.asarray([value]).astype(np.float32)
            for value in normalized_observation
        ], axis=-1)

        return low_dim_states_observation

    def _normalize(self, key, value):
        if key not in self._NORMALIZATION_VALUES:
            return value
        return value / self._NORMALIZATION_VALUES[key]


class SocialVehiclesObservationAdapter(ObservationAdapter):
    _RADIUS = 100.0
    _CAPACITY = 10  # Number of social vehicles we keep.
    _FEATURES = 4  # Number of features for each social vehicle.

    def __init__(self):
        super(SocialVehiclesObservationAdapter, self).__init__(
            name="social_vehicles",
            gym_space=gym.spaces.Box(
                low=-1e10,
                high=1e10,
                shape=(self._CAPACITY, self._FEATURES),
                dtype=np.ndarray,
            ),
            size=None,  # TODO: Fill this out.
            required_interfaces={
                "neighborhood_vehicles": NeighborhoodVehicles(self._RADIUS)
            }
        )

    def adapt(self, observation: Observation):
        ego_position = observation.ego_vehicle_state.position
        ego_heading = observation.ego_vehicle_state.heading
        social_vehicles = observation.neighborhood_vehicle_states

        # Sort by distance to the ego vehicle.
        social_vehicles.sort(
            key=lambda vehicle: self._get_distance(vehicle, ego_position), reverse=False
        )

        # Extract the state of each social vehicle.
        social_vehicles = np.asarray([
            self._extract_social_vehicle_state(
                social_vehicle=social_vehicle,
                ego_position=ego_position,
                ego_heading=ego_heading,
            )
            for social_vehicle in social_vehicles
        ], dtype=np.float32)

        # Pad with zero vectors if we don't have enough social vehicles.
        if len(social_vehicles) < self._CAPACITY:
            remain = self._CAPACITY - len(social_vehicles)
            empty_social_vehicles = np.zeros(
                shape=(remain, self._FEATURES), dtype=np.float32
            )
            social_vehicles = np.concatenate((social_vehicles, empty_social_vehicles))

        # Remove extra social vehicles if there were too many in the observation.
        social_vehicles = social_vehicles[:self._CAPACITY]

        return social_vehicles

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


class VectorObservationAdapter(ObservationAdapter):
    pass


class RewardAdapter:
    @property
    def required_interfaces(self):
        raise NotImplementedError

    def adapt(self, observation: Observation, reward):
        raise NotImplementedError


class DefaultRewardAdapter(RewardAdapter):
    # TODO: Have a required interfaces property.

    def adapt(self, observation: Observation, reward: float):
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
# required_interfaces = adapter.required_interfaces_from_types(
#     action_type=action_type,
#     observation_type=observation_type,
#     reward_type=reward,
# )
#
# spec = AgentSpec(
#     interface=AgentInterface(
#         **required_interfaces,
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
_TYPE_TO_ADAPTER = {
    ActionType.DISCRETE: None,  # TODO: Fill in.
    ActionType.CONTINUOUS: None,  # TODO: Fill in.
    ObservationType.VECTOR: VectorObservationAdapter,
    ObservationType.IMAGE: TopDownRgbObservationAdapter,
    RewardType.DEFAULT: DefaultRewardAdapter,
}


def action_space_from_type(action_type: Union[ActionType, str]) -> gym.Space:
    return _TYPE_TO_ADAPTER[action_type].gym_space


def action_adapter_from_type(action_type: Union[ActionType, str]) -> Callable:
    return _TYPE_TO_ADAPTER[action_type].adapt


def observation_space_from_type(
    observation_type: Union[ObservationType, str]
) -> gym.Space:
    return _TYPE_TO_ADAPTER[observation_type].gym_space


def observation_adapter_from_type(
    observation_type: Union[ObservationType, str]
) -> Callable:
    return _TYPE_TO_ADAPTER[observation_type].adapt


def reward_adapter_from_type(reward_type: Union[RewardAdapter, str]) -> Callable:
    return _TYPE_TO_ADAPTER[reward_type].adapt


def required_interfaces_from_types(
    action_type: Union[ActionType, str],
    observation_type: Union[ObservationType, str],
    reward_type: Union[RewardType, str],
) -> Dict[str, Any]:
    raise NotImplementedError


# NOTE: Idea for how the baseline adapter could work.
class BaselineAdapter:
    def __init__(self,
        policy_params,
        observation_adapters: Sequence[ObservationAdapter],
        reward_adapter: RewardAdapter,
    ):
        self._observation_adapters = observation_adapters
        self._reward_adapter = reward_adapter

    @classmethod
    def from_params(cls, params: Dict[str, Any]):
        assert "action_type" in params
        assert "observation_type" in params
        assert "reward_type" in params
        raise NotImplementedError

    @classmethod
    def from_type(cls, adapter_type: AdapterType):
        # TODO: Ensure adapter_type is valid.
        raise NotImplementedError

    # NOTE: Needed for RLlib.
    @property
    def observation_space(self):
        # return gym.spaces.Dict(
        #     {
        #         observation_adapter.name: observation_adapter.gym_space
        #         for observation_adapter in self._observation_adapters
        #     }
        # )
        return BaselineAdapter.observation_space_from_type(self._observation_type)

    # NOTE: Needed for RLlib.
    @property
    def action_space(self):
        # TODO: Change this so that it is either continuous or discrete.
        return gym.spaces.Box(
            low=np.array([0.0, 0.0, -1.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32,
        )

    @staticmethod
    def observation_space_from_type(observation_type: Union[ObservationType, str]):
        # To be used by agents when they need the size of the observation.
        if observation_type == ObservationType.VECTOR:
            # Return the observation description of the vector observation.
            observation_adapters = BaselineAdapter._observation_adapters_from_type(
                observation_type
            )
            return gym.spaces.Dict(
                {
                    observation_adapter.name: observation_adapter.gym_space
                    for observation_adapter in observation_adapters
                }
            )
        elif observation_type == ObservationType.IMAGE:
            # Return the observation description of the image observation.
            return gym.spaces.Dict(
                {
                    TopDownRgbObservationAdapter.name: TopDownRgbObservationAdapter.gym_space,
                }
            )
        else:
            raise Exception(f"Unsupported observation type {observation_type}.")

    @staticmethod
    def _observation_adapters_from_type(
        observation_type: Union[ObservationType, str]
    ) -> Sequence[ObservationAdapter]:
        if observation_type == ObservationType.VECTOR:
            return [LowDimStatesObservationAdapter, SocialVehiclesObservationAdapter]
        elif observation_type == ObservationType.IMAGE:
            return [TopDownRgbObservationAdapter]
        else:
            raise Exception(f"Unsupported observation type {observation_type}.")

    # def action_adapter(self, model_action):
    #     # print why this doesn't go through?
    #     throttle, brake, steering = model_action
    #     # print(M)
    #     return np.array([throttle, brake, steering * np.pi * 0.25])

    # XXX: Maybe not needed.
    def adapt_action(self, action):
        return action

    def adapt_observation(self, observation: Observation):
        adapted_observation = {
            observation_adapter.name: observation_adapter.adapt(observation)
            for observation_adapter in self._observation_adapters
        }
        return adapted_observation

    def adapt_reward(self, observation: Observation, reward: float):
        adapted_reward = self._reward_adapter(observation, reward)
        return adapted_reward

    # XXX: Maybe not needed.
    def adapt_info(self, observation: Observation, reward: float, info: Dict[str, Any]):
        return info
