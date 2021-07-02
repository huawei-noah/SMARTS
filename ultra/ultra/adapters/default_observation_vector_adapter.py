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
import copy

from scipy.spatial import distance
from typing import Dict, List, Tuple
import gym
import numpy as np

from smarts.core.coordinates import Heading
from smarts.core.sensors import Observation, VehicleObservation
from smarts.core.agent_interface import NeighborhoodVehicles, Waypoints
from ultra.adapters.constants import DEFAULT_RADIUS, DEFAULT_WAYPOINTS
from ultra.utils.common import get_closest_waypoint, get_path_to_goal, rotate2d_vector


_WAYPOINTS = DEFAULT_WAYPOINTS
_RADIUS = DEFAULT_RADIUS
_CAPACITY = 10  # Number of social vehicles we keep in the adapted observation.
_FEATURES = 4  # Number of features for each social vehicle.
_SIZE = (
    1  # Speed.
    + 1  # Distance from center.
    + 1  # Steering.
    + 1  # Angle error.
    + 2  # The x and y values of the goal's relative position.
    + 2 * _WAYPOINTS  # The x and y values of each waypoint's relative position.
    + 1  # Road speed.
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


# The space of the adapted observation.
gym_space: gym.Space = gym.spaces.Dict(
    {
        "low_dim_states": gym.spaces.Box(
            low=-1e10, high=1e10, shape=(_SIZE,), dtype=np.float32
        ),
        "social_vehicles": gym.spaces.Box(
            low=-1e10, high=1e10, shape=(_CAPACITY, _FEATURES), dtype=np.float32
        ),
    }
)
# This adapter requires SMARTS to pass the next _WAYPOINTS waypoints and all
# neighborhood vehicles within a radius of _RADIUS meters in the agent's observation.
required_interface = {
    "waypoints": Waypoints(lookahead=_WAYPOINTS),
    "neighborhood_vehicles": NeighborhoodVehicles(radius=_RADIUS),
}


def adapt(observation: Observation) -> Dict[str, np.ndarray]:
    """Adapts a raw environment observation into a dictionary of numpy.ndarrays.

    The raw observation from the environment must include the ego vehicle's state,
    events, waypoint paths, and neighborhood vehicles. See smarts.core.sensors for more
    information on the Observation type.

    Args:
        observation (Observation): The raw environment observation received from SMARTS.

    Returns:
        dict: A dictionary with two keys, "low_dim_states" and "social_vehicles". The
            value of "low_dim_states" is a numpy.ndarray with shape (_SIZE,), and the
            value of "social_vehicles" is a numpy.ndarray with shape
            (_CAPACITY, _FEATURES).
    """
    observation = copy.deepcopy(observation)
    ego_position = observation.ego_vehicle_state.position
    ego_heading = observation.ego_vehicle_state.heading
    ego_speed = observation.ego_vehicle_state.speed
    ego_steering = observation.ego_vehicle_state.steering
    ego_start = observation.ego_vehicle_state.mission.start
    ego_goal = observation.ego_vehicle_state.mission.goal
    ego_waypoints = observation.waypoint_paths

    ego_goal_path = get_path_to_goal(
        goal=ego_goal, paths=ego_waypoints, start=ego_start
    )
    ego_closest_waypoint, ego_lookahead_waypoints = get_closest_waypoint(
        num_lookahead=_WAYPOINTS,
        goal_path=ego_goal_path,
        ego_position=ego_position,
        ego_heading=ego_heading,
    )
    ego_lookahead_waypoints = np.hstack(ego_lookahead_waypoints)

    signed_distance_from_center = ego_closest_waypoint.signed_lateral_error(
        ego_position
    )
    lane_width = ego_closest_waypoint.lane_width * 0.5
    ego_distance_from_center = signed_distance_from_center / lane_width

    ego_relative_rotated_goal_position = rotate2d_vector(
        np.asarray(ego_goal.position[0:2]) - np.asarray(ego_position[0:2]),
        -ego_heading,
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
        waypoints_lookahead=ego_lookahead_waypoints,
    )
    normalized_observation = [
        _normalize(key, observation_dict[key]) for key in _NORMALIZATION_VALUES.keys()
    ]
    low_dim_states = np.concatenate(
        [
            value
            if isinstance(value, collections.abc.Iterable)
            else np.asarray([value])
            for value in normalized_observation
        ],
        axis=-1,
    )

    # Adapt the social vehicles.
    social_vehicles = observation.neighborhood_vehicle_states

    if len(social_vehicles) == 0:
        # There are no social vehicles. Create an empty array with the correct
        # number of features so it can be padded.
        social_vehicles = np.empty((0, _FEATURES))
    else:
        # Sort by distance to the ego vehicle.
        social_vehicles.sort(
            key=lambda vehicle: _get_distance(vehicle, ego_position),
            reverse=False,
        )
        # Extract the state of each social vehicle.
        social_vehicles = np.asarray(
            [
                _extract_social_vehicle_state(
                    social_vehicle=social_vehicle,
                    ego_position=ego_position,
                    ego_heading=ego_heading,
                )
                for social_vehicle in social_vehicles
            ]
        )

    if len(social_vehicles) < _CAPACITY:
        # Pad with zero vectors if we don't have enough social vehicles.
        remain = _CAPACITY - len(social_vehicles)
        empty_social_vehicles = np.zeros(shape=(remain, _FEATURES))
        social_vehicles = np.concatenate((social_vehicles, empty_social_vehicles))
    elif len(social_vehicles) > _CAPACITY:
        # Remove extra social vehicles if there were too many in the observation.
        social_vehicles = social_vehicles[:_CAPACITY]

    vector_observation = {
        "low_dim_states": low_dim_states.astype(np.float32),
        "social_vehicles": social_vehicles.astype(np.float32),
    }
    return vector_observation


def _normalize(key, value):
    if key not in _NORMALIZATION_VALUES:
        return value
    return value / _NORMALIZATION_VALUES[key]


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
        social_vehicle.speed / 30.0,
    ]
    return social_vehicle_state


def _get_distance(
    social_vehicle: VehicleObservation, ego_position: Tuple[float, float, float]
) -> float:
    return distance.euclidean(social_vehicle.position[:2], ego_position[:2])
