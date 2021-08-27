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
import math

import numpy as np
from scipy.spatial import distance

from smarts.core.agent_interface import Waypoints
from smarts.core.sensors import Observation
from ultra.adapters.constants import DEFAULT_WAYPOINTS
from ultra.utils.common import get_closest_waypoint, get_path_to_goal


_WAYPOINTS = DEFAULT_WAYPOINTS


# This adapter requires SMARTS to pass the next _WAYPOINTS waypoints in the agent's
# observation.
required_interface = {"waypoints": Waypoints(lookahead=_WAYPOINTS)}


def adapt(observation: Observation, reward: float) -> float:
    """Adapts a raw environment observation and an environment reward to a custom reward
    of type float.

    The raw observation from the environment must include the ego vehicle's state,
    events, and waypoint paths. See smarts.core.sensors for more information on the
    Observation type.

    Args:
        observation (Observation): The raw environment observation received from SMARTS.
        reward (float): The environment reward received from SMARTS.

    Returns:
        float: The adapted, custom reward which includes aspects of the ego vehicle's
            state and the ego vehicle's mission progress, in addition to the environment
            reward.
    """
    env_reward = reward
    ego_events = observation.events
    ego_observation = observation.ego_vehicle_state
    start = observation.ego_vehicle_state.mission.start
    goal = observation.ego_vehicle_state.mission.goal
    path = get_path_to_goal(goal=goal, paths=observation.waypoint_paths, start=start)

    linear_jerk = np.linalg.norm(ego_observation.linear_jerk)
    angular_jerk = np.linalg.norm(ego_observation.angular_jerk)

    closest_wp, _ = get_closest_waypoint(
        num_lookahead=_WAYPOINTS,
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

    speed_fraction = max(0, ego_observation.speed / closest_wp.speed_limit)

    # NOTE: This requires the NeighborhoodVehicles interface.
    # number of violations
    # (ego_num_violations, social_num_violations,) = ego_social_safety(
    #     observation,
    #     d_min_ego=1.0,
    #     t_c_ego=1.0,
    #     d_min_social=1.0,
    #     t_c_social=1.0,
    #     ignore_vehicle_behind=True,
    # )

    # Environment reward from SMARTS
    env_reward /= 100

    # Termination reward
    ego_reached_goal_reward = 1.0 if ego_events.reached_goal else 0.0

    # Termination reward
    ego_collision_reward = -1.0 if len(ego_events.collisions) > 0 else 0.0
    ego_off_road_reward = -1.0 if ego_events.off_road else 0.0
    ego_off_route_reward = -1.0 if ego_events.off_route else 0.0
    ego_wrong_way_reward = -1.0 if ego_events.wrong_way else 0.0

    # Intermediate rewards/penalties per step
    ego_step_reward = 0.02 * min(speed_fraction, 1) * np.cos(angle_error)
    ego_dist_center_reward = -0.005 * min(1, abs(ego_dist_center))
    ego_angle_error_reward = 0.005 * max(0, np.cos(angle_error))
    ego_linear_jerk = -0.0001 * linear_jerk
    ego_angular_jerk = -0.0001 * angular_jerk * math.cos(angle_error)

    # Speed reward
    if speed_fraction < 0.01:  # Speed below threshold
        ego_speed_reward = -0.01
    elif speed_fraction >= 1:  # Speed above limit
        ego_speed_reward = -0.1
    else:
        ego_speed_reward = 0.0

    # NOTE: This requires the NeighborhoodVehicles interface.
    # ego_safety_reward = -0.02 if ego_num_violations > 0 else 0
    # NOTE: This requires the NeighborhoodVehicles interface.
    # social_safety_reward = -0.02 if social_num_violations > 0 else 0

    rewards = sum(
        [
            ego_reached_goal_reward,
            ego_collision_reward,
            ego_off_road_reward,
            ego_off_route_reward,
            ego_wrong_way_reward,
            ego_speed_reward,
            ego_dist_center_reward,
            ego_angle_error_reward,
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
