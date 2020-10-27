# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
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
from typing import Callable
from dataclasses import dataclass

import numpy as np
import gym

from smarts.core.utils.math import vec_2d, vec_to_radians, squared_dist
from smarts.core.coordinates import Heading


@dataclass
class Adapter:
    space: gym.Space
    transform: Callable


def scan_for_vehicle(
    target_prefix,
    angle_a,
    angle_b,
    activation_dist_squared,
    self_vehicle_state,
    other_vehicle_state,
):
    if target_prefix and not other_vehicle_state.id.startswith(target_prefix):
        return False

    min_angle, max_angle = min(angle_a, angle_b), max(angle_a, angle_b)
    sqd = squared_dist(self_vehicle_state.position, other_vehicle_state.position)
    # check for activation distance
    if sqd < activation_dist_squared:
        direction = np.sum(
            [other_vehicle_state.position, -self_vehicle_state.position], axis=0
        )
        angle = Heading(vec_to_radians(direction[:2]))
        rel_angle = angle.relative_to(self_vehicle_state.heading)
        return min_angle <= rel_angle <= max_angle
    return False


def scan_for_vehicles(
    target_prefix,
    angle_a,
    angle_b,
    activation_dist_squared,
    self_vehicle_state,
    other_vehicle_states,
    short_circuit: bool = False,
):
    if target_prefix:
        other_vehicle_states = filter(
            lambda v: v.id.startswith(target_prefix), other_vehicle_states
        )

    min_angle, max_angle = min(angle_a, angle_b), max(angle_a, angle_b)
    vehicles = []

    for vehicle_state in other_vehicle_states:
        sqd = squared_dist(self_vehicle_state.position, vehicle_state.position)
        # check for activation distance
        if sqd < activation_dist_squared:
            direction = np.sum(
                [vehicle_state.position, -self_vehicle_state.position], axis=0
            )
            angle = Heading(vec_to_radians(direction[:2]))
            rel_angle = angle.relative_to(self_vehicle_state.heading)
            if min_angle <= rel_angle <= max_angle:
                vehicles.append(vehicle_state)
                if short_circuit:
                    break
    return vehicles


_LANE_TTC_OBSERVATION_SPACE = gym.spaces.Dict(
    {
        "distance_from_center": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
        "angle_error": gym.spaces.Box(low=-np.pi, high=np.pi, shape=(1,)),
        "speed": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
        "steering": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
        "ego_lane_dist": gym.spaces.Box(low=-1e10, high=1e10, shape=(3,)),
        "ego_ttc": gym.spaces.Box(low=-1e10, high=1e10, shape=(3,)),
    }
)


def _lane_ttc_observation_adapter(env_observation):
    ego = env_observation.ego_vehicle_state
    waypoint_paths = env_observation.waypoint_paths
    wps = [path[0] for path in waypoint_paths]

    # distance of vehicle from center of lane
    closest_wp = min(wps, key=lambda wp: wp.dist_to(ego.position))
    signed_dist_from_center = closest_wp.signed_lateral_error(ego.position)
    lane_hwidth = closest_wp.lane_width * 0.5
    norm_dist_from_center = signed_dist_from_center / lane_hwidth

    ego_ttc, ego_lane_dist = _ego_ttc_lane_dist(env_observation, closest_wp.lane_index)

    return {
        "distance_from_center": np.array([norm_dist_from_center]),
        "angle_error": np.array([closest_wp.relative_heading(ego.heading)]),
        "speed": np.array([ego.speed]),
        "steering": np.array([ego.steering]),
        "ego_ttc": np.array(ego_ttc),
        "ego_lane_dist": np.array(ego_lane_dist),
    }


lane_ttc_observation_adapter = Adapter(
    space=_LANE_TTC_OBSERVATION_SPACE, transform=_lane_ttc_observation_adapter
)


def _ego_ttc_lane_dist(env_observation, ego_lane_index):
    ttc_by_p, lane_dist_by_p = _ttc_by_path(env_observation)

    return _ego_ttc_calc(ego_lane_index, ttc_by_p, lane_dist_by_p)


def _ttc_by_path(env_observation):
    ego = env_observation.ego_vehicle_state
    waypoint_paths = env_observation.waypoint_paths
    neighborhood_vehicle_states = env_observation.neighborhood_vehicle_states

    # first sum up the distance between waypoints along a path
    # ie. [(wp1, path1, 0),
    #      (wp2, path1, 0 + dist(wp1, wp2)),
    #      (wp3, path1, 0 + dist(wp1, wp2) + dist(wp2, wp3))]

    wps_with_lane_dist = []
    for path_idx, path in enumerate(waypoint_paths):
        lane_dist = 0.0
        for w1, w2 in zip(path, path[1:]):
            wps_with_lane_dist.append((w1, path_idx, lane_dist))
            lane_dist += np.linalg.norm(w2.pos - w1.pos)
        wps_with_lane_dist.append((path[-1], path_idx, lane_dist))

    # next we compute the TTC along each of the paths
    ttc_by_path_index = [1000] * len(waypoint_paths)
    lane_dist_by_path_index = [1] * len(waypoint_paths)

    for v in neighborhood_vehicle_states:
        # find all waypoints that are on the same lane as this vehicle
        wps_on_lane = [
            (wp, path_idx, dist)
            for wp, path_idx, dist in wps_with_lane_dist
            if wp.lane_id == v.lane_id
        ]

        if not wps_on_lane:
            # this vehicle is not on a nearby lane
            continue

        # find the closest waypoint on this lane to this vehicle
        nearest_wp, path_idx, lane_dist = min(
            wps_on_lane, key=lambda tup: np.linalg.norm(tup[0].pos - vec_2d(v.position))
        )

        if np.linalg.norm(nearest_wp.pos - vec_2d(v.position)) > 2:
            # this vehicle is not close enough to the path, this can happen
            # if the vehicle is behind the ego, or ahead past the end of
            # the waypoints
            continue

        relative_speed_m_per_s = (ego.speed - v.speed) * 1000 / 3600
        if abs(relative_speed_m_per_s) < 1e-5:
            relative_speed_m_per_s = 1e-5

        ttc = lane_dist / relative_speed_m_per_s
        ttc /= 10
        if ttc <= 0:
            # discard collisions that would have happened in the past
            continue

        lane_dist /= 100
        lane_dist_by_path_index[path_idx] = min(
            lane_dist_by_path_index[path_idx], lane_dist
        )
        ttc_by_path_index[path_idx] = min(ttc_by_path_index[path_idx], ttc)

    return ttc_by_path_index, lane_dist_by_path_index


def _ego_ttc_calc(ego_lane_index, ttc_by_path, lane_dist_by_path):
    ego_ttc = [0] * 3
    ego_lane_dist = [0] * 3

    ego_ttc[1] = ttc_by_path[ego_lane_index]
    ego_lane_dist[1] = lane_dist_by_path[ego_lane_index]

    max_lane_index = len(ttc_by_path) - 1
    min_lane_index = 0
    if ego_lane_index + 1 > max_lane_index:
        ego_ttc[2] = 0
        ego_lane_dist[2] = 0
    else:
        ego_ttc[2] = ttc_by_path[ego_lane_index + 1]
        ego_lane_dist[2] = lane_dist_by_path[ego_lane_index + 1]
    if ego_lane_index - 1 < min_lane_index:
        ego_ttc[0] = 0
        ego_lane_dist[0] = 0
    else:
        ego_ttc[0] = ttc_by_path[ego_lane_index - 1]
        ego_lane_dist[0] = lane_dist_by_path[ego_lane_index - 1]
    return ego_ttc, ego_lane_dist
