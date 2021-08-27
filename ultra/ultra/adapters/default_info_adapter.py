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
from typing import Any, Dict

import numpy as np
from scipy.spatial import distance

from smarts.core.agent_interface import NeighborhoodVehicles, Waypoints
from smarts.core.sensors import Observation
from ultra.adapters.constants import DEFAULT_RADIUS, DEFAULT_WAYPOINTS
import ultra.adapters.default_reward_adapter as default_reward_adapter
from ultra.utils.common import ego_social_safety, get_closest_waypoint, get_path_to_goal


_WAYPOINTS = DEFAULT_WAYPOINTS
_RADIUS = DEFAULT_RADIUS


required_interface = {
    "waypoints": Waypoints(lookahead=_WAYPOINTS),
    "neighborhood_vehicles": NeighborhoodVehicles(radius=_RADIUS),
}


def adapt(
    observation: Observation, reward: float, info: Dict[str, Any]
) -> Dict[str, Any]:
    """Adapts a raw environment observation, an environment reward, and info about the
    agent from the environment into custom information about the agent.

    The raw observation from the environment must include the ego vehicle's state,
    events, waypoint paths, and neighborhood vehicles. See smarts.core.sensors for more
    information on the Observation type.

    Args:
        observation (Observation): The raw environment observation received from SMARTS.
        reward (float): The environment reward received from SMARTS.
        info (dict): Information about the agent received from SMARTS.

    Returns:
        dict: The adapted information. A dictionary containing the same information as
            the original info argument, but also including a "logs" key containing more
            information about the agent.
    """
    ego_state = observation.ego_vehicle_state
    start = observation.ego_vehicle_state.mission.start
    goal = observation.ego_vehicle_state.mission.goal
    path = get_path_to_goal(goal=goal, paths=observation.waypoint_paths, start=start)
    closest_wp, _ = get_closest_waypoint(
        num_lookahead=100,
        goal_path=path,
        ego_position=ego_state.position,
        ego_heading=ego_state.heading,
    )
    signed_dist_from_center = closest_wp.signed_lateral_error(ego_state.position)
    lane_width = closest_wp.lane_width * 0.5
    ego_dist_center = signed_dist_from_center / lane_width

    linear_jerk = np.linalg.norm(ego_state.linear_jerk)
    angular_jerk = np.linalg.norm(ego_state.angular_jerk)

    # Distance to goal
    ego_2d_position = ego_state.position[0:2]
    goal_dist = distance.euclidean(ego_2d_position, goal.position[0:2])

    angle_error = closest_wp.relative_heading(
        ego_state.heading
    )  # relative heading radians [-pi, pi]

    # number of violations
    (ego_num_violations, social_num_violations,) = ego_social_safety(
        observation,
        d_min_ego=1.0,
        t_c_ego=1.0,
        d_min_social=1.0,
        t_c_social=1.0,
        ignore_vehicle_behind=True,
    )

    info["logs"] = dict(
        position=ego_state.position,
        speed=ego_state.speed,
        steering=ego_state.steering,
        heading=ego_state.heading,
        dist_center=abs(ego_dist_center),
        start=start,
        goal=goal,
        closest_wp=closest_wp,
        events=observation.events,
        ego_num_violations=ego_num_violations,
        social_num_violations=social_num_violations,
        goal_dist=goal_dist,
        linear_jerk=np.linalg.norm(ego_state.linear_jerk),
        angular_jerk=np.linalg.norm(ego_state.angular_jerk),
        env_score=default_reward_adapter.adapt(observation, reward),
    )

    return info
