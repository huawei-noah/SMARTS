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
import numpy as np
import random
from scipy.spatial.distance import euclidean
from collections import defaultdict

from ultra.utils.common import rotate2d_vector


HORIZONTAL = 0
VERTICAL = 1
REAR = 2
FRONT = 3


def extract_social_vehicle_state_default(
    social_vehicle, ego_vehicle_pos, ego_vehicle_heading, social_vehicle_config=None
):
    social_vehicle_position = social_vehicle.position[0:2] - ego_vehicle_pos[0:2]
    social_vehicle_position_rotated = rotate2d_vector(
        social_vehicle_position, -ego_vehicle_heading
    )
    return [
        (social_vehicle_position_rotated[0]) / 100.0,
        (social_vehicle_position_rotated[1]) / 100.0,
        (social_vehicle.heading - ego_vehicle_heading) / 3.14,
        social_vehicle.speed / 30.0,
    ]


def extract_social_vehicle_state_pointnet(
    social_vehicle, ego_vehicle_pos, ego_vehicle_heading, social_vehicle_config
):
    speed_norm = 30
    heading_diff = social_vehicle.heading - ego_vehicle_heading
    # vector length is prop to the speed
    heading_vector = np.asarray([np.cos(heading_diff), np.sin(heading_diff)]) * (
        social_vehicle.speed / speed_norm
    )
    social_vehicle_position = social_vehicle.position[0:2] - ego_vehicle_pos[0:2]
    social_vehicle_position_rotated = rotate2d_vector(
        social_vehicle_position, -ego_vehicle_heading
    )
    return [
        (social_vehicle_position_rotated[0]) / 100.0,
        (social_vehicle_position_rotated[1]) / 100.0,
        heading_vector[0],
        heading_vector[1],
    ]


def get_social_vehicles(
    ego_vehicle_pos,
    ego_vehicle_heading,
    neighborhood_vehicles,
    social_vehicle_config,
    waypoint_paths,
):
    extractor_func = social_vehicle_config["social_vehicle_extractor_func"]
    encoder_key = social_vehicle_config["encoder_key"]
    if social_vehicle_config["encoder"]["use_leading_vehicles"]:
        social_vehicles = get_social_vehicles_leading(
            ego_vehicle_pos=ego_vehicle_pos,
            ego_vehicle_heading=ego_vehicle_heading,
            neighborhood_vehicles=neighborhood_vehicles,
            waypoint_paths=waypoint_paths,
            extractor_func=extractor_func,
            **social_vehicle_config["encoder"]["use_leading_vehicles"]
        )
    else:
        social_vehicles = get_social_vehicles_states_sorted_by_distance(
            ego_vehicle_pos=ego_vehicle_pos,
            ego_vehicle_heading=ego_vehicle_heading,
            neighborhood_vehicles=neighborhood_vehicles,
            social_vehicle_config=social_vehicle_config,
            extractor_func=extractor_func,
        )
    social_vehicles = np.asarray(social_vehicles).astype(np.float32)
    return social_vehicles


def get_social_vehicles_states_sorted_by_distance(
    ego_vehicle_pos,
    ego_vehicle_heading,
    neighborhood_vehicles,
    social_vehicle_config,
    extractor_func,
):
    social_vehicles = neighborhood_vehicles
    social_vehicles.sort(
        key=lambda vehicle: get_distance(vehicle, ego_vehicle_pos), reverse=False
    )
    social_vehicles = [
        extractor_func(
            social_vehicle=vehicle,
            ego_vehicle_pos=ego_vehicle_pos,
            ego_vehicle_heading=ego_vehicle_heading,
            social_vehicle_config=social_vehicle_config,
        )
        for vehicle in social_vehicles
    ]
    return social_vehicles


def get_social_vehicles_leading(
    ego_vehicle_pos,
    ego_vehicle_heading,
    neighborhood_vehicles,
    extractor_func,
    max_dist_social_vehicle,
    num_social_vehicle_per_lane,
    social_capacity,
    waypoint_paths,
):
    social_vehicles, vehicles_on_ego_lane = [], []
    vehicles_on_other_lanes = defaultdict(lambda: [])

    # find ego lane information
    ego_wps = [path[0] for path in waypoint_paths]
    ego_closest_waypoint, prev_ego_wp = None, None
    ego_closest_waypoint_min_dist = float("inf")
    for i, wp in enumerate(ego_wps):
        temp_dist = wp.dist_to(ego_vehicle_pos)
        if temp_dist < ego_closest_waypoint_min_dist:
            prev_ego_wp = ego_wps[max(i - 1, 0)]
            ego_closest_waypoint = wp
            ego_closest_waypoint_min_dist = temp_dist
    ego_lane_id = ego_closest_waypoint.lane_id
    ego_lane_index = ego_closest_waypoint.lane_index

    # if vehicle is within the max_dist consider:
    #   1- find vehicles on ego lane , and only keep the closest front and rear vehicles
    #   2- find vehicles on other lanes
    for vehicle in neighborhood_vehicles:
        dist = get_distance(vehicle, ego_vehicle_pos)
        if dist < max_dist_social_vehicle:
            if vehicle.lane_id == ego_lane_id and vehicle.lane_index == ego_lane_index:
                vehicles_on_ego_lane.append(vehicle)
            else:
                vehicles_on_other_lanes[(vehicle.lane_id, vehicle.lane_index)].append(
                    vehicle
                )

    # process vehicles on ego lane and only keep the nearest front/rear
    if len(vehicles_on_ego_lane) > 0:
        front_dist, rear_dist = float("inf"), float("inf")
        front_vehicle, rear_vehicle = None, None
        for vehicle in vehicles_on_ego_lane:
            location, _ = find_relative_location(
                vehicle.position, ego_vehicle_pos, ego_vehicle_heading
            )
            dist = get_distance(vehicle, ego_vehicle_pos)
            if location == FRONT and dist < front_dist:
                front_vehicle = vehicle
                front_dist = dist
            elif location == REAR and dist < rear_dist:
                rear_vehicle = vehicle
                rear_dist = dist
        if front_vehicle:
            social_vehicles.append(
                extractor_func(
                    social_vehicle=front_vehicle,
                    ego_vehicle_pos=ego_vehicle_pos,
                    ego_vehicle_heading=ego_vehicle_heading,
                )
            )
        if rear_vehicle:
            social_vehicles.append(
                extractor_func(
                    social_vehicle=rear_vehicle,
                    ego_vehicle_pos=ego_vehicle_pos,
                    ego_vehicle_heading=ego_vehicle_heading,
                )
            )

    # process vehicles on the other lanes:
    # sort vehicles based on their distance per lane and keep num_social_vehicle_per_lane
    for lane, vehicles in vehicles_on_other_lanes.items():
        vehicles_on_other_lanes[lane].sort(
            key=lambda vehicle: get_distance(vehicle, ego_vehicle_pos)
        )
        vehicles_on_other_lanes[lane] = vehicles_on_other_lanes[lane][
            : min(len(vehicles_on_other_lanes), num_social_vehicle_per_lane)
        ]
        for v in vehicles_on_other_lanes[lane]:
            social_vehicles.append(
                extractor_func(v, ego_vehicle_pos, ego_vehicle_heading)
            )
    # add fake social vehicles if necessary
    if len(social_vehicles) < social_capacity:
        # find a waypoint on ego path equal to max_dist
        fake_social_vehicle = make_fake_social_vehicle(
            ego_vehicle_pos, ego_vehicle_heading, prev_ego_wp, max_dist_social_vehicle
        )
        fake_social_vehicles = [
            fake_social_vehicle for i in range(social_capacity - len(social_vehicles))
        ]
        social_vehicles.extend(fake_social_vehicles)

    # Permute the social vehicles randomly
    social_vehicles_permuted = np.random.permutation(social_vehicles)
    # If we have more vehicles than required, then delete the tail
    if len(social_vehicles_permuted) > social_capacity:
        social_vehicles_permuted = social_vehicles_permuted[:social_capacity]

    return social_vehicles_permuted


def make_fake_social_vehicle(
    ego_vehicle_pos, ego_vehicle_heading, prev_ego_wp, max_dist_social_vehicle
):
    # DG: I changed to zeros because I think it'll be easier for learning
    return [0.0, 0.0, 0.0, 0.0]


"""
def make_fake_social_vehicle(ego_vehicle_pos, ego_vehicle_heading, prev_ego_wp, max_dist_social_vehicle):
    fake_social_vehicle = [
        ego_vehicle_pos[0],
        ego_vehicle_pos[1],
        0.0,
        0.0,
    ]
    location, orientation = find_relative_location(prev_ego_wp.pos, ego_vehicle_pos, ego_vehicle_heading)
    if orientation == VERTICAL:
        if location == REAR:
            fake_social_vehicle[1] -= max_dist_social_vehicle
        else:
            fake_social_vehicle[1] += max_dist_social_vehicle
    else:
        if location == REAR:
            fake_social_vehicle[0] += max_dist_social_vehicle
        else:
            fake_social_vehicle[0] -= max_dist_social_vehicle
    fake_social_vehicle[0] /= 100.0
    fake_social_vehicle[1] /= 100.0
    return fake_social_vehicle
"""


def find_relative_location(position, target_position, heading):
    location = None
    orientation = None
    x_diff = position[0] - target_position[0]
    y_diff = position[1] - target_position[1]

    # DG: Rotate to get relative location - This won't work due to assumptions in this code about orientation
    # rotated_diff = rotate2d_vector(np.array([x_diff, y_diff]), -heading)
    # x_diff, y_diff = rotated_diff[0], rotated_diff[1]

    # TODO: DG - I think there is a bug here since it assumes a road orientation.
    #   We probably need to
    #    1) compute the direction vector of the closest waypoint
    #    2) find the normal of the waypoint direction
    #    3) determine the side that the waypoint is on via a projection onto the normal
    #   We don't need orientation so we can just return FRONT versus REAR
    #   We won't know which is which (FRONT versus REAR) with this method but we can distinguish by the sign.
    #   One will be positive and one will be negative and we want to return the social vehicles in each lane
    #   with the smallest projection for each sign (positive and negative signs).
    #   Details: https://math.stackexchange.com/questions/274712/calculate-on-which-side-of-a-straight-line-is-a-given-point-located
    if abs(x_diff) > abs(y_diff):
        orientation = HORIZONTAL
        location = REAR if x_diff < 0 else FRONT
    else:
        orientation = VERTICAL
        location = FRONT if y_diff > 0 else REAR
    return location, orientation


def get_distance(social_vehicle, ego_vehicle_pos):
    return euclidean(social_vehicle.position[:2], ego_vehicle_pos[:2])
