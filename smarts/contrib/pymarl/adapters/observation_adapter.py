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
import math

import gym
import numpy as np

from smarts.core.utils.math import vec_2d, sign
from smarts.core.coordinates import Heading


DEFAULT_OBSERVATION_SPACE = gym.spaces.Dict(
    {
        "distance_from_center": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
        "speed": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
        "steering": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
        "ego_lane_dist": gym.spaces.Box(low=-1e10, high=1e10, shape=(5,)),
        "ego_ttc": gym.spaces.Box(low=-1e10, high=1e10, shape=(5,)),
        # treated as a boolean
        "ego_will_crash": gym.spaces.Box(low=0, high=1, dtype=np.int8, shape=(1,)),
        "speed_of_closest": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
        "proximity": gym.spaces.Box(low=-1e10, high=1e10, shape=(6,)),
        "headings_of_cars": gym.spaces.Box(low=-1e10, high=1e10, shape=(3,)),
    }
)


def default_obs_adapter(env_obs):
    ego = env_obs.ego_vehicle_state
    waypoint_paths = env_obs.waypoint_paths
    wps = [path[0] for path in waypoint_paths]

    # Distance of vehicle from center of lane
    closest_wp = min(wps, key=lambda wp: wp.dist_to(ego.position))
    signed_dist_from_center = closest_wp.signed_lateral_error(ego.position)
    lane_hwidth = closest_wp.lane_width * 0.5
    norm_dist_from_center = signed_dist_from_center / lane_hwidth

    (
        ttc_by_path_,
        lane_dist_by_path,
        headings_of_cars_all_lanes,
        speed_of_closest,
        will_crash,
    ) = ttc_by_path(
        ego, waypoint_paths, env_obs.neighborhood_vehicle_states, closest_wp
    )

    ego_lane_index = closest_wp.lane_index
    ego_ttc, ego_lane_dist = ego_ttc_calc(
        ego, ego_lane_index, ttc_by_path_, lane_dist_by_path
    )

    ego_ttc = np.array(ego_ttc)
    ego_lane_dist = np.array(ego_lane_dist)

    # XXX: Model was trained on max 3 lanes, remove extra lane observations if lane > 3
    # for safety reasons (can be fixed by training on +3 lane environments).
    if sum(ego_ttc > 0) > 3:
        ego_ttc[0] = 0
        ego_ttc[-1] = 0
    if sum(ego_lane_dist > 0) > 3:
        ego_lane_dist[0] = 0
        ego_lane_dist[-1] = 0
    if env_obs.occupancy_grid_map is not None:
        proximity = proximity_detection(env_obs.occupancy_grid_map[1])
    else:
        proximity = [0] * 6

    headings_of_cars = [0] * 3
    # current lane is centre
    headings_of_cars[1] = headings_of_cars_all_lanes[ego_lane_index]
    if headings_of_cars[1] == 0:
        headings_of_cars[1] = headings_of_cars_all_lanes[ego_lane_index]
    if ego_lane_index + 1 > len(headings_of_cars_all_lanes) - 1:
        headings_of_cars[2] = 0
    else:
        headings_of_cars[2] = headings_of_cars_all_lanes[ego_lane_index + 1]

    if ego_lane_index - 1 < 0:
        headings_of_cars[0] = 0
    else:
        headings_of_cars[0] = headings_of_cars_all_lanes[ego_lane_index - 1]

    obs = {
        "distance_from_center": np.array([norm_dist_from_center]),
        "speed": np.array([ego.speed / 120]),
        "steering": np.array([ego.steering]),
        "ego_lane_dist": np.array(ego_lane_dist),
        "ego_ttc": np.array(ego_ttc),
        "ego_will_crash": np.array([will_crash]),
        "speed_of_closest": np.array([speed_of_closest]),
        "proximity": proximity,
        "headings_of_cars": np.array(headings_of_cars),
    }

    return obs


def proximity_detection(occupancy_grid_map):
    """Detects other vehicles in the vicinity of the ego vehicle."""
    boxes = [
        occupancy_grid_map[11:25, 23:27],  # front left
        occupancy_grid_map[11:25, 27:37],  # front centre
        occupancy_grid_map[11:25, 37:41],  # front right
        occupancy_grid_map[25:39, 23:27],  # left
        occupancy_grid_map[25:39, 37:41],  # right
        occupancy_grid_map[39:53, 27:37],  # back
    ]
    return np.array([b.max() > 0 for b in boxes], np.float32)


# Fix taken from:
# https://gist.github.com/davidrusu/d144a2646c9597a0d412c7e795ada548#file-nv_heading_to_ego_heading-py
def nv_heading_to_ego_heading(nv_heading):
    heading = nv_heading + math.pi * 0.5
    if heading < 0:
        heading += 2 * math.pi
    return Heading(heading)


def ttc_by_path(ego, waypoint_paths, neighborhood_vehicle_states, ego_closest_wp):
    # TODO: Phase this out; used for penalizing close proximity to other cars
    will_crash = False

    wps_with_lane_dist = []
    for path_idx, path in enumerate(waypoint_paths):
        lane_dist = 0.0
        for w1, w2 in zip(path, path[1:]):
            wps_with_lane_dist.append((w1, path_idx, lane_dist))
            lane_dist += np.linalg.norm(w2.pos - w1.pos)
        wps_with_lane_dist.append((path[-1], path_idx, lane_dist))

    # next we compute the TTC along each of the paths
    ttc_by_path_index = [1] * len(waypoint_paths)
    lane_dist_by_path_index = [1] * len(waypoint_paths)
    headings_of_cars = [0] * len(waypoint_paths)

    speed_of_closest = 1
    wps = [path[0] for path in waypoint_paths]
    ego_closest_wp = min(wps, key=lambda wp: wp.dist_to(ego.position))
    neighborhood_vehicle_states = neighborhood_vehicle_states or []
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
            wps_on_lane,
            key=lambda tup: np.linalg.norm(tup[0].pos - vec_2d(v.position)),
        )

        if np.linalg.norm(nearest_wp.pos - vec_2d(v.position)) > 2:
            # this vehicle is not close enough to the path, this can happen
            # if the vehicle is behind the ego, or ahead past the end of
            # the waypoints
            continue

        if ego_closest_wp.lane_index == nearest_wp.lane_index:
            if np.linalg.norm(vec_2d(ego.position) - vec_2d(v.position)) < 6:
                will_crash = True

        relative_speed_m_per_s = (ego.speed - v.speed) * 1000 / 3600
        if abs(relative_speed_m_per_s) < 1e-5:
            relative_speed_m_per_s = 1e-5
        dist_wp_vehicle_vector = vec_2d(v.position) - vec_2d(nearest_wp.pos)
        # take into account the position of the car instead of its nearest waypoint
        direction_vector = np.array(
            [math.cos(nearest_wp.heading), math.sin(nearest_wp.heading),]
        ).dot(dist_wp_vehicle_vector)
        dist_to_vehicle = lane_dist + sign(direction_vector) * (
            np.linalg.norm(vec_2d(nearest_wp.pos) - vec_2d(v.position))
        )
        ttc = dist_to_vehicle / relative_speed_m_per_s
        ttc = ttc / 10
        lane_dist = dist_to_vehicle / 100

        if lane_dist_by_path_index[path_idx] > lane_dist:
            if nearest_wp.lane_index == v.lane_index:
                headings_of_cars[path_idx] = math.sin(
                    nearest_wp.relative_heading(nv_heading_to_ego_heading(v.heading))
                )

            # speed
            if ego_closest_wp.lane_index == v.lane_index:
                speed_of_closest = (v.speed - ego.speed) / 120

        lane_dist_by_path_index[path_idx] = min(
            lane_dist_by_path_index[path_idx], lane_dist
        )

        if ttc <= 0:
            # discard collisions that would have happened in the past
            continue

        ttc_by_path_index[path_idx] = min(ttc_by_path_index[path_idx], ttc)

    return (
        ttc_by_path_index,
        lane_dist_by_path_index,
        headings_of_cars,
        speed_of_closest,
        will_crash,
    )


# original function extended to support 5 lanes
def ego_ttc_calc(ego, ego_lane_index, ttc_by_path, lane_dist_by_path):
    # ttc, lane distance from ego perspective
    ego_ttc = [0] * 5
    ego_lane_dist = [0] * 5

    # current lane is centre
    ego_ttc[2] = ttc_by_path[ego_lane_index]
    ego_lane_dist[2] = lane_dist_by_path[ego_lane_index]

    max_lane_index = len(ttc_by_path) - 1
    min_lane_index = 0
    if ego_lane_index + 1 > max_lane_index:
        ego_ttc[3] = 0
        ego_lane_dist[3] = 0
        ego_ttc[4] = 0
        ego_lane_dist[4] = 0
    elif ego_lane_index + 2 > max_lane_index:
        ego_ttc[3] = ttc_by_path[ego_lane_index + 1]
        ego_lane_dist[3] = lane_dist_by_path[ego_lane_index + 1]
        ego_ttc[4] = 0
        ego_lane_dist[4] = 0
    else:
        ego_ttc[3] = ttc_by_path[ego_lane_index + 1]
        ego_lane_dist[3] = lane_dist_by_path[ego_lane_index + 1]
        ego_ttc[4] = ttc_by_path[ego_lane_index + 2]
        ego_lane_dist[4] = lane_dist_by_path[ego_lane_index + 2]

    if ego_lane_index - 1 < min_lane_index:
        ego_ttc[0] = 0
        ego_lane_dist[0] = 0
        ego_ttc[1] = 0
        ego_lane_dist[1] = 0
    elif ego_lane_index - 2 < min_lane_index:
        ego_ttc[0] = 0
        ego_lane_dist[0] = 0
        ego_ttc[1] = ttc_by_path[ego_lane_index - 1]
        ego_lane_dist[1] = lane_dist_by_path[ego_lane_index - 1]
    else:
        ego_ttc[0] = ttc_by_path[ego_lane_index - 2]
        ego_lane_dist[0] = lane_dist_by_path[ego_lane_index - 2]
        ego_ttc[1] = ttc_by_path[ego_lane_index - 1]
        ego_lane_dist[1] = lane_dist_by_path[ego_lane_index - 1]
    return ego_ttc, ego_lane_dist
