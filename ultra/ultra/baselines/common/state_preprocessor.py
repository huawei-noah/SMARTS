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
from typing import (
    List,
    Tuple,
)
import numpy as np

from smarts.core.scenario import (
    Goal,
    Start,
)
from smarts.core.sensors import (
    Heading,
    Observation,
)
from ultra.baselines.common.social_vehicle_extraction import (
    get_social_vehicles_leading,
    get_social_vehicles_states_sorted_by_distance,
)
from ultra.utils.common import (
    get_closest_waypoint,
    get_path_to_goal,
)


class StatePreprocessor:
    def __call__(self, state: Observation, *args, **kwargs):
        return self._preprocess_state(state, *args, **kwargs)

    def _preprocess_state(self, state: Observation, *args, **kwargs):
        raise NotImplementedError("State preprocessing method is not defined.")

    @staticmethod
    def get_state_description(*args, **kwargs):
        raise NotImplementedError("State description is not defined.")

    @staticmethod
    def get_social_vehicles_representation(
        social_vehicles,
        social_vehicle_config,
        ego_position,
        ego_heading,
        ego_waypoints,
    ):
        if social_vehicle_config["encoder"]["use_leading_vehicles"]:
            social_vehicles = get_social_vehicles_leading(
                ego_vehicle_pos=ego_position,
                ego_vehicle_heading=ego_heading,
                neighborhood_vehicles=social_vehicles,
                waypoint_paths=ego_waypoints,
                extractor_func=social_vehicle_config["social_vehicle_extractor_func"],
                **social_vehicle_config["encoder"]["use_leading_vehicles"],
            )
        else:
            social_vehicles = get_social_vehicles_states_sorted_by_distance(
                ego_vehicle_pos=ego_position,
                ego_vehicle_heading=ego_heading,
                neighborhood_vehicles=social_vehicles,
                social_vehicle_config=social_vehicle_config,
                extractor_func=social_vehicle_config["social_vehicle_extractor_func"],
            )
        social_vehicles = np.asarray(social_vehicles).astype(np.float32)
        return social_vehicles

    @staticmethod
    def extract_ego_speed(state: Observation) -> float:
        """
        Extracts the ego vehicle's speed from a raw environment observation.
            Parameters:
                state (Observation): The raw environment observation.
            Returns:
                (float): The ego vehicle's speed in meters per second.
        """
        return state.ego_vehicle_state.speed

    @staticmethod
    def extract_ego_steering(state: Observation) -> float:
        """
        Extracts the ego vehicle's speed from a raw environment observation.
            Parameters:
                state (Observation): The raw environment observation.
            Returns:
                (float): The angle of the front wheels in radians.
        """
        return state.ego_vehicle_state.steering

    @staticmethod
    def extract_ego_position(state: Observation) -> Tuple:
        """
        Extracts the ego vehicle's position from a raw environment observation.
            Parameters:
                state (Observation): The raw environment observation.
            Returns:
                (numpy.array): A 3D NumPy array of the x, y, and z coordinates of
                    the center of the vehicle's bounding box's bottom plane.
        """
        return state.ego_vehicle_state.position

    @staticmethod
    def extract_ego_heading(state: Observation) -> Heading:
        """
        Extracts the ego vehicle's heading from a raw environment observation.
            Parameters:
                state (Observation): The raw environment observation.
            Returns:
                (Heading): A class inheriting from float, containing the vehicle's
                    heading in radians.
        """
        return state.ego_vehicle_state.heading

    @staticmethod
    def extract_ego_waypoints(state: Observation) -> List:
        """
        Extracts the ego vehicle's waypoints from a raw environment observation.
            Parameters:
                state (Observation): The raw environment observation.
            Returns:
                (List[List[Waypoint]]): A list of a list of the ego vehicle's waypoints.
        """
        return state.waypoint_paths

    @staticmethod
    def extract_ego_start(state: Observation) -> Start:
        """
        Extracts the ego vehicle's mission start from a raw environemnt observation.
            Parameters:
                state (Observation): The raw environment observation.
            Returns:
                (Start): A dataclass containing a 'position' attribute containing the
                    x and y coordinates of the start position in a tuple, and a 'heading'
                    attribute containing the Heading of the start position.
        """
        return state.ego_vehicle_state.mission.start

    @staticmethod
    def extract_ego_goal(state: Observation) -> Goal:
        """
        Extracts the ego vehicle's mission goal from a raw environment observation.
            Parameters:
                state (Observation): The raw environment observation.
            Returns:
                (PositionalGoal): A dataclass containing a 'position' attribute containing
                    the x and y coordinates of the goal in a tuple, and a float 'radius'
                    attribute containing the radius in meters the vehicle has to be in
                    order to have reached the goal
        """
        return state.ego_vehicle_state.mission.goal

    @staticmethod
    def extract_ego_goal_path(ego_goal, ego_waypoints, ego_start):
        return get_path_to_goal(goal=ego_goal, paths=ego_waypoints, start=ego_start)

    @staticmethod
    def extract_closest_waypoint(
        ego_goal_path, ego_position, ego_heading, num_lookahead=100
    ):
        return get_closest_waypoint(
            num_lookahead=num_lookahead,
            goal_path=ego_goal_path,
            ego_position=ego_position,
            ego_heading=ego_heading,
        )

    @staticmethod
    def extract_social_vehicles(state: Observation) -> List:
        """
        Extracts the social vehicles from a raw environment observation.
            Parameters:
                state (Observation): The raw environment observation.
            Returns:
                (List[VehicleObservation]): A list of vehicle observations containing
                    data about each neighboring social vehicle.
        """
        return state.neighborhood_vehicle_states
