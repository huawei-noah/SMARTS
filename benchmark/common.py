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
from collections import defaultdict
from typing import Dict, Sequence

import cv2
import gym
import numpy as np
from ray import logger
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy

from smarts.core.controllers import ActionSpaceType
from smarts.core.scenario import PositionalGoal
from smarts.core.sensors import Observation
from smarts.core.utils.math import vec_2d

SPACE_LIB = dict(
    # normalized distance to lane center
    distance_to_center=lambda _: gym.spaces.Box(
        low=-float("inf"), high=float("inf"), shape=(1,)
    ),
    heading_errors=lambda look: gym.spaces.Box(
        low=-float("inf"),
        high=float("inf"),
        shape=(look[0],),
    ),
    speed=lambda _: gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(1,)),
    steering=lambda _: gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(1,)),
    goal_relative_pos=lambda _: gym.spaces.Box(
        low=-float("inf"), high=float("inf"), shape=(2,)
    ),
    neighbor=lambda neighbor_num: gym.spaces.Box(
        low=-float("inf"),
        high=float("inf"),
        shape=(neighbor_num * 5,),
    ),
    img_gray=lambda shape: gym.spaces.Box(
        low=-float("inf"), high=float("inf"), shape=shape
    ),
    lane_its_info=lambda _: gym.spaces.Box(
        low=-float("inf"), high=float("inf"), shape=(16,)
    ),
    # To discover micro information around ego car in 16*16m ogm
    # proximity array around ego car
    proximity=lambda _: gym.spaces.Box(low=-1e10, high=1e10, shape=(8,)),
)


def _cal_angle(vec):

    if vec[1] < 0:
        base_angle = math.pi
        base_vec = np.array([-1.0, 0.0])
    else:
        base_angle = 0.0
        base_vec = np.array([1.0, 0.0])

    cos = vec.dot(base_vec) / np.sqrt(vec.dot(vec) + base_vec.dot(base_vec))
    angle = math.acos(cos)
    return angle + base_angle


def _get_closest_vehicles(ego, neighbor_vehicles, n):
    ego_pos = ego.position[:2]
    groups = {i: (None, 1e10) for i in range(n)}
    partition_size = math.pi * 2.0 / n
    # get partition
    for v in neighbor_vehicles:
        v_pos = v.position[:2]
        rel_pos_vec = np.asarray([v_pos[0] - ego_pos[0], v_pos[1] - ego_pos[1]])
        # calculate its partitions
        angle = _cal_angle(rel_pos_vec)
        i = int(angle / partition_size)
        dist = np.sqrt(rel_pos_vec.dot(rel_pos_vec))
        if dist < groups[i][1]:
            groups[i] = (v, dist)

    return groups


def proximity_detection(OGM):
    """
    Detects other vehicles in the vicinity of the ego vehicle
    hard coded for OGM(64, 64, 0.25)
    """
    boxes = []
    boxes += [
        OGM[11:25, 23:27],  # front left
        OGM[11:25, 27:37],  # front center
        OGM[11:25, 37:41],  # front right
        OGM[25:39, 23:27],  # left
        OGM[25:39, 37:41],  # right
        OGM[41:53, 23:27],  # back left
        OGM[41:53, 27:37],  # back center
        OGM[41:53, 37:41],  # back right
    ]
    output = np.array([b.max() > 0 for b in boxes], np.float32)
    return output


def heading_to_degree(heading):
    # +y = 0 rad. Note the 0 means up direction
    return np.degrees(heading % (2 * math.pi))


def heading_to_vec(heading):
    # axis x: right, y:up
    angle = (heading + math.pi * 0.5) % (2 * math.pi)
    return np.array([math.cos(angle), math.sin(angle)])


def trans_ego_center(ego_lane_index, origin_info):
    # transform lane ttc and dist to make ego lane in the array center
    assert len(origin_info) == 5

    # index need to be set to zero
    # 4: [0,1], 3:[0], 2:[], 1:[4], 0:[3,4]
    zero_index = [[3, 4], [4], [], [0], [0, 1]]
    zero_index = zero_index[ego_lane_index]

    origin_info[zero_index] = 0
    new_info = np.roll(origin_info, 2 - ego_lane_index)

    return new_info


class ActionSpace:
    @staticmethod
    def from_type(space_type):
        if space_type == ActionSpaceType.Continuous:
            return gym.spaces.Box(
                low=np.array([0.0, 0.0, -1.0]),
                high=np.array([1.0, 1.0, 1.0]),
                dtype=np.float32,
            )
        elif space_type == ActionSpaceType.Lane:
            return gym.spaces.Discrete(4)
        else:
            raise NotImplementedError


lane_crash_flag = False
intersection_crash_flag = False


class CalObs:
    @staticmethod
    def cal_goal_relative_pos(env_obs: Observation, _):
        """ Return normalized relative position (2-dimensional). """

        ego_state = env_obs.ego_vehicle_state
        goal = ego_state.mission.goal
        assert isinstance(goal, PositionalGoal), goal

        ego_pos = ego_state.position[:2]
        goal_pos = goal.position  # the position of mission goal is 2-dimensional.
        vector = np.asarray([goal_pos[0] - ego_pos[0], goal_pos[1] - ego_pos[1]])
        # space = SPACE_LIB["goal_relative_pos"](None)
        # return vector / (space.high - space.low)
        return vector

    @staticmethod
    def cal_distance_to_center(env_obs: Observation, _):
        """ Calculate the signed distance to the center of the current lane. """

        ego = env_obs.ego_vehicle_state
        waypoint_paths = env_obs.waypoint_paths
        wps = [path[0] for path in waypoint_paths]
        closest_wp = min(wps, key=lambda wp: wp.dist_to(ego.position))
        signed_dist_to_center = closest_wp.signed_lateral_error(ego.position)
        # lane_hwidth = closest_wp.lane_width * 0.5
        # norm_dist_from_center = signed_dist_to_center / lane_hwidth

        # dist = np.asarray([norm_dist_from_center])
        dist = np.asarray([signed_dist_to_center])
        return dist

    @staticmethod
    def cal_heading_errors(env_obs: Observation, *args):
        look_ahead, look_type = args
        ego = env_obs.ego_vehicle_state
        waypoint_paths = env_obs.waypoint_paths
        wps = [path[0] for path in waypoint_paths]
        closest_wp = min(wps, key=lambda wp: wp.dist_to(ego.position))
        closest_path = waypoint_paths[closest_wp.lane_index]
        closest_path_len = len(closest_path)

        if look_type == "continuous":
            wp_indices = np.arange(look_ahead)
        else:
            wp_indices = np.array([0, 1, 2, 3, 5, 8, 13, 21, 34, 50])[:look_ahead]

        first_larger_value = np.argmax(wp_indices > closest_path_len - 1)
        if first_larger_value == 0:
            pass
        else:
            wp_indices[first_larger_value:] = wp_indices[first_larger_value - 1]

        closest_path_wps = [closest_path[i] for i in wp_indices]

        heading_errors = [
            math.sin(math.radians(wp.relative_heading(ego.heading)))
            for wp in closest_path_wps
        ]

        return np.asarray(heading_errors)

    @staticmethod
    def cal_speed(env_obs: Observation, _):
        ego = env_obs.ego_vehicle_state
        res = np.asarray([ego.speed])
        # return res * 3.6 / 120
        return res * 3.6

    @staticmethod
    def cal_steering(env_obs: Observation, _):
        ego = env_obs.ego_vehicle_state
        return np.asarray([ego.steering / (0.5 * math.pi)])

    @staticmethod
    def cal_neighbor(env_obs: Observation, closest_neighbor_num):
        ego = env_obs.ego_vehicle_state
        neighbor_vehicle_states = env_obs.neighborhood_vehicle_states
        # dist, speed, ttc, pos
        features = np.zeros((closest_neighbor_num, 5))
        # fill neighbor vehicles into closest_neighboor_num areas
        surrounding_vehicles = _get_closest_vehicles(
            ego, neighbor_vehicle_states, n=closest_neighbor_num
        )

        heading_angle = ego.heading + math.pi / 2.0
        ego_heading_vec = np.asarray([math.cos(heading_angle), math.sin(heading_angle)])
        for i, v in surrounding_vehicles.items():
            if v[0] is None:
                continue
            v = v[0]
            rel_pos = np.asarray(
                list(map(lambda x: x[0] - x[1], zip(v.position[:2], ego.position[:2])))
            )

            rel_dist = np.sqrt(rel_pos.dot(rel_pos))

            v_heading_angle = math.radians(v.heading)
            v_heading_vec = np.asarray(
                [math.cos(v_heading_angle), math.sin(v_heading_angle)]
            )

            ego_heading_norm_2 = ego_heading_vec.dot(ego_heading_vec)
            rel_pos_norm_2 = rel_pos.dot(rel_pos)
            v_heading_norm_2 = v_heading_vec.dot(v_heading_vec)

            ego_cosin = ego_heading_vec.dot(rel_pos) / np.sqrt(
                ego_heading_norm_2 + rel_pos_norm_2
            )

            v_cosin = v_heading_vec.dot(rel_pos) / np.sqrt(
                v_heading_norm_2 + rel_pos_norm_2
            )

            rel_speed = 0
            if ego_cosin <= 0 and v_cosin > 0:
                rel_speed = 0
            else:
                rel_speed = ego.speed * ego_cosin - v.speed * v_cosin

            ttc = min(rel_dist / max(1e-5, rel_speed), 5.0)

            features[i, :] = np.asarray(
                [rel_dist, rel_speed, ttc, rel_pos[0], rel_pos[1]]
            )

        return features.reshape((-1,))

    @staticmethod
    def cal_ego_lane_dist_and_speed(env_obs: Observation, observe_lane_num):
        """Calculate the distance from ego vehicle to its front vehicles (if have) at observed lanes,
        also the relative speed of the front vehicle which positioned at the same lane.
        """
        ego = env_obs.ego_vehicle_state
        waypoint_paths = env_obs.waypoint_paths
        wps = [path[0] for path in waypoint_paths]
        closest_wp = min(wps, key=lambda wp: wp.dist_to(ego.position))

        wps_with_lane_dist = []
        for path_idx, path in enumerate(waypoint_paths):
            lane_dist = 0.0
            for w1, w2 in zip(path, path[1:]):
                wps_with_lane_dist.append((w1, path_idx, lane_dist))
                lane_dist += np.linalg.norm(w2.pos - w1.pos)
            wps_with_lane_dist.append((path[-1], path_idx, lane_dist))

        # TTC calculation along each path
        ego_closest_wp = min(wps, key=lambda wp: wp.dist_to(ego.position))

        wps_on_lane = [
            (wp, path_idx, dist)
            for wp, path_idx, dist in wps_with_lane_dist
            # if wp.lane_id == v.lane_id
        ]

        ego_lane_index = closest_wp.lane_index
        lane_dist_by_path = [1] * len(waypoint_paths)
        ego_lane_dist = [0] * observe_lane_num
        speed_of_closest = 0.0

        for v in env_obs.neighborhood_vehicle_states:
            nearest_wp, path_idx, lane_dist = min(
                wps_on_lane,
                key=lambda tup: np.linalg.norm(tup[0].pos - vec_2d(v.position)),
            )
            if np.linalg.norm(nearest_wp.pos - vec_2d(v.position)) > 2:
                # this vehicle is not close enough to the path, this can happen
                # if the vehicle is behind the ego, or ahead past the end of
                # the waypoints
                continue

            # relative_speed_m_per_s = (ego.speed - v.speed) * 1000 / 3600
            # relative_speed_m_per_s = max(abs(relative_speed_m_per_s), 1e-5)
            dist_wp_vehicle_vector = vec_2d(v.position) - vec_2d(nearest_wp.pos)
            direction_vector = np.array(
                [
                    math.cos(math.radians(nearest_wp.heading)),
                    math.sin(math.radians(nearest_wp.heading)),
                ]
            ).dot(dist_wp_vehicle_vector)

            dist_to_vehicle = lane_dist + np.sign(direction_vector) * (
                np.linalg.norm(vec_2d(nearest_wp.pos) - vec_2d(v.position))
            )
            lane_dist = dist_to_vehicle / 100.0

            if lane_dist_by_path[path_idx] > lane_dist:
                if ego_closest_wp.lane_index == v.lane_index:
                    speed_of_closest = (v.speed - ego.speed) / 120.0

            lane_dist_by_path[path_idx] = min(lane_dist_by_path[path_idx], lane_dist)

        # current lane is centre
        flag = observe_lane_num // 2
        ego_lane_dist[flag] = lane_dist_by_path[ego_lane_index]

        max_lane_index = len(lane_dist_by_path) - 1

        if max_lane_index == 0:
            right_sign, left_sign = 0, 0
        else:
            right_sign = -1 if ego_lane_index + 1 > max_lane_index else 1
            left_sign = -1 if ego_lane_index - 1 >= 0 else 1

        ego_lane_dist[flag + right_sign] = lane_dist_by_path[
            ego_lane_index + right_sign
        ]
        ego_lane_dist[flag + left_sign] = lane_dist_by_path[ego_lane_index + left_sign]

        res = np.asarray(ego_lane_dist + [speed_of_closest])
        return res

    @staticmethod
    def cal_lane_its_info(env_obs: Observation, _):
        """
        cal neighbour info includes lane info and intersection info
        """
        # init flag, dist, ttc, headings
        global lane_crash_flag
        global intersection_crash_flag
        lane_crash_flag = False
        intersection_crash_flag = False

        # default 10s
        lane_ttc = np.array([1] * 5, dtype=float)
        # default 100m
        lane_dist = np.array([1] * 5, dtype=float)
        # default 120km/h
        closest_lane_nv_rel_speed = 1

        intersection_ttc = 1
        intersection_distance = 1
        closest_its_nv_rel_speed = 1
        # default 100m
        closest_its_nv_rel_pos = np.array([1, 1])

        wp_paths = env_obs.waypoint_paths
        ego = env_obs.ego_vehicle_state
        neighborhood_vehicle_states = env_obs.neighborhood_vehicle_states
        closest_wps = [path[0] for path in wp_paths]

        # distance of vehicle from center of lane
        ego_closest_wp = min(closest_wps, key=lambda wp: wp.dist_to(ego.position))

        ego_lane_index = ego_closest_wp.lane_index

        # here to set invalid value to 0
        wp_paths_num = len(wp_paths)
        lane_ttc[wp_paths_num:] = 0
        lane_dist[wp_paths_num:] = 0

        features = np.concatenate(
            [
                trans_ego_center(ego_lane_index, lane_ttc),
                trans_ego_center(ego_lane_index, lane_dist),
                [
                    closest_lane_nv_rel_speed,
                    intersection_ttc,
                    intersection_distance,
                    closest_its_nv_rel_speed,
                ],
                closest_its_nv_rel_pos,
            ]
        )
        # return if no neighbour vehicle or off the routes(no waypoint paths)
        if not neighborhood_vehicle_states or not wp_paths_num:
            return features
        # merge waypoint paths (consider might not the same length)
        merge_waypoint_paths = []
        for wp_path in wp_paths:
            merge_waypoint_paths += wp_path

        wp_poses = np.array([wp.pos for wp in merge_waypoint_paths])

        # compute neighbour vehicle closest wp
        nv_poses = np.array([nv.position for nv in neighborhood_vehicle_states])
        nv_wp_distance = np.linalg.norm(
            nv_poses[:, :2][:, np.newaxis] - wp_poses, axis=2
        )
        nv_closest_wp_index = np.argmin(nv_wp_distance, axis=1)
        nv_closest_distance = np.min(nv_wp_distance, axis=1)

        # get not in same lane id social vehicles(intersect vehicles and behind vehicles)
        wp_lane_ids = np.array([wp.lane_id for wp in merge_waypoint_paths])
        nv_lane_ids = np.array([nv.lane_id for nv in neighborhood_vehicle_states])
        not_in_same_lane_id = nv_lane_ids[:, np.newaxis] != wp_lane_ids
        not_in_same_lane_id = np.all(not_in_same_lane_id, axis=1)

        ego_edge_id = ego.lane_id[1:-2] if ego.lane_id[0] == "-" else ego.lane_id[:-2]
        nv_edge_ids = np.array(
            [
                nv.lane_id[1:-2] if nv.lane_id[0] == "-" else nv.lane_id[:-2]
                for nv in neighborhood_vehicle_states
            ]
        )
        not_in_ego_edge_id = nv_edge_ids[:, np.newaxis] != ego_edge_id
        not_in_ego_edge_id = np.squeeze(not_in_ego_edge_id, axis=1)

        is_not_closed_nv = not_in_same_lane_id & not_in_ego_edge_id
        not_closed_nv_index = np.where(is_not_closed_nv)[0]

        # filter sv not close to the waypoints including behind the ego or ahead past the end of the waypoints
        close_nv_index = np.where(nv_closest_distance < 2)[0]

        if not close_nv_index.size:
            pass
        else:
            close_nv = [neighborhood_vehicle_states[i] for i in close_nv_index]

            # calculate waypoints distance to ego car along the routes
            wps_with_lane_dist_list = []
            for wp_path in wp_paths:
                path_wp_poses = np.array([wp.pos for wp in wp_path])
                wp_poses_shift = np.roll(path_wp_poses, 1, axis=0)
                wps_with_lane_dist = np.linalg.norm(
                    path_wp_poses - wp_poses_shift, axis=1
                )
                wps_with_lane_dist[0] = 0
                wps_with_lane_dist = np.cumsum(wps_with_lane_dist)
                wps_with_lane_dist_list += wps_with_lane_dist.tolist()
            wps_with_lane_dist_list = np.array(wps_with_lane_dist_list)

            # get neighbour vehicle closest waypoints index
            nv_closest_wp_index = nv_closest_wp_index[close_nv_index]
            # ego car and neighbour car distance, not very accurate since use the closest wp
            ego_nv_distance = wps_with_lane_dist_list[nv_closest_wp_index]

            # get neighbour vehicle lane index
            nv_lane_index = np.array(
                [merge_waypoint_paths[i].lane_index for i in nv_closest_wp_index]
            )

            # get wp path lane index
            lane_index_list = [wp_path[0].lane_index for wp_path in wp_paths]

            for i, lane_index in enumerate(lane_index_list):
                # get same lane vehicle
                same_lane_nv_index = np.where(nv_lane_index == lane_index)[0]
                if not same_lane_nv_index.size:
                    continue
                same_lane_nv_distance = ego_nv_distance[same_lane_nv_index]
                closest_nv_index = same_lane_nv_index[np.argmin(same_lane_nv_distance)]
                closest_nv = close_nv[closest_nv_index]
                closest_nv_speed = closest_nv.speed
                closest_nv_heading = closest_nv.heading
                # radius to degree
                closest_nv_heading = heading_to_degree(closest_nv_heading)

                closest_nv_pos = closest_nv.position[:2]
                bounding_box = closest_nv.bounding_box

                # map the heading to make it consistent with the position coordination
                map_heading = (closest_nv_heading + 90) % 360
                map_heading_radius = np.radians(map_heading)
                nv_heading_vec = np.array(
                    [np.cos(map_heading_radius), np.sin(map_heading_radius)]
                )
                nv_heading_vertical_vec = np.array(
                    [-nv_heading_vec[1], nv_heading_vec[0]]
                )

                # get four edge center position (consider one vehicle take over two lanes when change lane)
                # maybe not necessary
                closest_nv_front = closest_nv_pos + bounding_box.length * nv_heading_vec
                closest_nv_behind = (
                    closest_nv_pos - bounding_box.length * nv_heading_vec
                )
                closest_nv_left = (
                    closest_nv_pos + bounding_box.width * nv_heading_vertical_vec
                )
                closest_nv_right = (
                    closest_nv_pos - bounding_box.width * nv_heading_vertical_vec
                )
                edge_points = np.array(
                    [
                        closest_nv_front,
                        closest_nv_behind,
                        closest_nv_left,
                        closest_nv_right,
                    ]
                )

                ep_wp_distance = np.linalg.norm(
                    edge_points[:, np.newaxis] - wp_poses, axis=2
                )
                ep_closed_wp_index = np.argmin(ep_wp_distance, axis=1)
                ep_closed_wp_lane_index = set(
                    [merge_waypoint_paths[i].lane_index for i in ep_closed_wp_index]
                    + [lane_index]
                )

                min_distance = np.min(same_lane_nv_distance)

                if ego_closest_wp.lane_index in ep_closed_wp_lane_index:
                    if min_distance < 6:
                        lane_crash_flag = True

                    nv_wp_heading = (
                        closest_nv_heading
                        - heading_to_degree(
                            merge_waypoint_paths[
                                nv_closest_wp_index[closest_nv_index]
                            ].heading
                        )
                    ) % 360

                    # find those car just get from intersection lane into ego lane
                    if nv_wp_heading > 30 and nv_wp_heading < 330:
                        relative_close_nv_heading = (
                            closest_nv_heading - heading_to_degree(ego.heading)
                        )
                        # map nv speed to ego car heading
                        map_close_nv_speed = closest_nv_speed * np.cos(
                            np.radians(relative_close_nv_heading)
                        )
                        closest_lane_nv_rel_speed = min(
                            closest_lane_nv_rel_speed,
                            (map_close_nv_speed - ego.speed) * 3.6 / 120,
                        )
                    else:
                        closest_lane_nv_rel_speed = min(
                            closest_lane_nv_rel_speed,
                            (closest_nv_speed - ego.speed) * 3.6 / 120,
                        )

                relative_speed_m_per_s = ego.speed - closest_nv_speed

                if abs(relative_speed_m_per_s) < 1e-5:
                    relative_speed_m_per_s = 1e-5

                ttc = min_distance / relative_speed_m_per_s
                # normalized into 10s
                ttc /= 10

                for j in ep_closed_wp_lane_index:
                    if min_distance / 100 < lane_dist[j]:
                        # normalize into 100m
                        lane_dist[j] = min_distance / 100

                    if ttc <= 0:
                        continue

                    if j == ego_closest_wp.lane_index:
                        if ttc < 0.1:
                            lane_crash_flag = True

                    if ttc < lane_ttc[j]:
                        lane_ttc[j] = ttc

        # get vehicles not in the waypoints lane
        if not not_closed_nv_index.size:
            pass
        else:
            filter_nv = [neighborhood_vehicle_states[i] for i in not_closed_nv_index]

            nv_pos = np.array([nv.position for nv in filter_nv])[:, :2]
            nv_heading = heading_to_degree(np.array([nv.heading for nv in filter_nv]))
            nv_speed = np.array([nv.speed for nv in filter_nv])

            ego_pos = ego.position[:2]
            ego_heading = heading_to_degree(ego.heading)
            ego_speed = ego.speed
            nv_to_ego_vec = nv_pos - ego_pos

            line_heading = (
                (np.arctan2(nv_to_ego_vec[:, 1], nv_to_ego_vec[:, 0]) * 180 / np.pi)
                - 90
            ) % 360
            nv_to_line_heading = (nv_heading - line_heading) % 360
            ego_to_line_heading = (ego_heading - line_heading) % 360

            # judge two heading whether will intersect
            same_region = (nv_to_line_heading - 180) * (
                ego_to_line_heading - 180
            ) > 0  # both right of line or left of line
            ego_to_nv_heading = ego_to_line_heading - nv_to_line_heading
            valid_relative_angle = (
                (nv_to_line_heading - 180 > 0) & (ego_to_nv_heading > 0)
            ) | ((nv_to_line_heading - 180 < 0) & (ego_to_nv_heading < 0))

            # emit behind vehicles
            valid_intersect_angle = np.abs(line_heading - ego_heading) < 90

            # emit patient vehicles which stay in the intersection
            not_patient_nv = nv_speed > 0.01

            # get valid intersection sv
            intersect_sv_index = np.where(
                same_region
                & valid_relative_angle
                & valid_intersect_angle
                & not_patient_nv
            )[0]

            if not intersect_sv_index.size:
                pass
            else:
                its_nv_pos = nv_pos[intersect_sv_index][:, :2]
                its_nv_speed = nv_speed[intersect_sv_index]
                its_nv_to_line_heading = nv_to_line_heading[intersect_sv_index]
                line_heading = line_heading[intersect_sv_index]
                # ego_to_line_heading = ego_to_line_heading[intersect_sv_index]

                # get intersection closest vehicle
                ego_nv_distance = np.linalg.norm(its_nv_pos - ego_pos, axis=1)
                ego_closest_its_nv_index = np.argmin(ego_nv_distance)
                ego_closest_its_nv_distance = ego_nv_distance[ego_closest_its_nv_index]

                line_heading = line_heading[ego_closest_its_nv_index]
                ego_to_line_heading = (
                    heading_to_degree(ego_closest_wp.heading) - line_heading
                ) % 360

                ego_closest_its_nv_speed = its_nv_speed[ego_closest_its_nv_index]
                its_closest_nv_to_line_heading = its_nv_to_line_heading[
                    ego_closest_its_nv_index
                ]
                # rel speed along ego-nv line
                closest_nv_rel_speed = ego_speed * np.cos(
                    np.radians(ego_to_line_heading)
                ) - ego_closest_its_nv_speed * np.cos(
                    np.radians(its_closest_nv_to_line_heading)
                )
                closest_nv_rel_speed_m_s = closest_nv_rel_speed
                if abs(closest_nv_rel_speed_m_s) < 1e-5:
                    closest_nv_rel_speed_m_s = 1e-5
                ttc = ego_closest_its_nv_distance / closest_nv_rel_speed_m_s

                intersection_ttc = min(intersection_ttc, ttc / 10)
                intersection_distance = min(
                    intersection_distance, ego_closest_its_nv_distance / 100
                )

                # transform relative pos to ego car heading coordinate
                rotate_axis_angle = np.radians(90 - ego_to_line_heading)
                closest_its_nv_rel_pos = (
                    np.array(
                        [
                            ego_closest_its_nv_distance * np.cos(rotate_axis_angle),
                            ego_closest_its_nv_distance * np.sin(rotate_axis_angle),
                        ]
                    )
                    / 100
                )

                closest_its_nv_rel_speed = min(
                    closest_its_nv_rel_speed, -closest_nv_rel_speed * 3.6 / 120
                )

                if ttc < 0:
                    pass
                else:
                    intersection_ttc = min(intersection_ttc, ttc / 10)
                    intersection_distance = min(
                        intersection_distance, ego_closest_its_nv_distance / 100
                    )

                    # if to collide in 2s or its distance in 6, make it slow down
                    if ttc < 2 or ego_closest_its_nv_distance < 6:
                        intersection_crash_flag = True

        features = np.concatenate(
            [
                trans_ego_center(ego_lane_index, lane_ttc),
                trans_ego_center(ego_lane_index, lane_dist),
                [
                    closest_lane_nv_rel_speed,
                    intersection_ttc,
                    intersection_distance,
                    closest_its_nv_rel_speed,
                ],
                closest_its_nv_rel_pos,
            ]
        )

        return features

    @staticmethod
    def cal_img_gray(env_obs: Observation, *args):
        # args = (height, width)
        resize = args

        def rgb2gray(rgb):
            return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

        rgb_ndarray = env_obs.top_down_rgb.data
        gray_scale = (
            cv2.resize(
                rgb2gray(rgb_ndarray), dsize=resize, interpolation=cv2.INTER_CUBIC
            )
            / 255.0
        )
        return gray_scale

    @staticmethod
    def cal_proximity(env_obs: Observation, _):
        proximity = proximity_detection(env_obs.occupancy_grid_map[1])
        return proximity


class SimpleCallbacks(DefaultCallbacks):
    """See example from (>=0.8.6): https://github.com/ray-project/ray/blob/master/rllib/examples
    /custom_metrics_and_callbacks.py"""

    def on_episode_start(
        self,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: MultiAgentEpisode,
        **kwargs,
    ):
        logger.info("episode {} started".format(episode.episode_id))
        episode.user_data["ego_speed"] = defaultdict(lambda: [])
        episode.user_data["step_heading_error"] = dict()

    def on_episode_step(
        self,
        worker: RolloutWorker,
        base_env: BaseEnv,
        episode: MultiAgentEpisode,
        **kwargs,
    ):
        ego_speed = episode.user_data["ego_speed"]
        for agent_id, obs in episode._agent_to_last_raw_obs.items():
            if isinstance(obs, list):
                obs = obs[-1]  # keep the lastest frame
            if isinstance(obs, dict):
                ego_speed[agent_id].append(obs.get("speed", -1.0))

    def on_episode_end(
        self,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: MultiAgentEpisode,
        **kwargs,
    ):
        ego_speed = episode.user_data["ego_speed"]
        mean_ego_speed = {
            agent_id: np.mean(speed_hist) for agent_id, speed_hist in ego_speed.items()
        }

        distance_travelled = dict()
        for _id, info in episode._agent_to_last_info.items():
            if info.get("_group_info"):
                for i, _info in enumerate(info["_group_info"]):
                    distance_travelled[f"{_id}:AGENT-{i}"] = np.mean(_info["score"])
            else:
                distance_travelled[_id] = np.mean(info["score"])

        speed_list = list(map(lambda x: round(x, 3), mean_ego_speed.values()))
        dist_list = list(map(lambda x: round(x, 3), distance_travelled.values()))
        reward_list = list(map(lambda x: round(x, 3), episode.agent_rewards.values()))

        episode.custom_metrics[f"mean_ego_speed"] = sum(speed_list) / max(
            1, len(speed_list)
        )
        episode.custom_metrics[f"distance_travelled"] = sum(dist_list) / max(
            1, len(dist_list)
        )

        logger.info(f"episode {episode.episode_id} ended with {episode.length} steps")


class ActionAdapter:
    @staticmethod
    def from_type(space_type):
        if space_type == ActionSpaceType.Continuous:
            return ActionAdapter.continuous_action_adapter
        elif space_type == ActionSpaceType.Lane:
            return ActionAdapter.discrete_action_adapter
        else:
            raise NotImplementedError

    @staticmethod
    def continuous_action_adapter(policy_action):
        assert len(policy_action) == 3
        return np.asarray(policy_action)

    @staticmethod
    def discrete_action_adapter(policy_action):
        if isinstance(policy_action, (list, tuple, np.ndarray)):
            action = np.argmax(policy_action)
        else:
            action = policy_action

        if action == 0:
            return "keep_lane"
        elif action == 1:
            return "slow_down"
        elif action == 2:
            return "change_lane_left"
        elif action == 3:
            return "change_lane_right"


def subscribe_features(**kwargs):
    res = dict()

    for k, config in kwargs.items():
        if bool(config):
            res[k] = SPACE_LIB[k](config)

    return res


def cal_obs(env_obs, space, feature_configs):
    if isinstance(space, gym.spaces.Dict):
        obs_np = {}
        for name in space.spaces:
            if hasattr(CalObs, f"cal_{name}"):
                args = (
                    (feature_configs[name],)
                    if not isinstance(feature_configs[name], Sequence)
                    else feature_configs[name]
                )
            obs_np[name] = getattr(CalObs, f"cal_{name}")(env_obs, *args)
    elif isinstance(space, gym.spaces.Tuple):
        obs_np = []
        assert isinstance(env_obs, Sequence)
        for obs, sub_space in zip(env_obs, space.spaces):
            obs_np.append(cal_obs(obs, sub_space, feature_configs))
    else:
        raise TypeError(f"Unexpected space type={type(space)}")
    return obs_np


def get_distance_from_center(env_obs):
    ego_state = env_obs.ego_vehicle_state
    wp_paths = env_obs.waypoint_paths
    closest_wps = [path[0] for path in wp_paths]

    # distance of vehicle from center of lane
    closest_wp = min(closest_wps, key=lambda wp: wp.dist_to(ego_state.position))
    signed_dist_from_center = closest_wp.signed_lateral_error(ego_state.position)
    lane_hwidth = closest_wp.lane_width * 0.5
    norm_dist_from_center = signed_dist_from_center / lane_hwidth

    return norm_dist_from_center
