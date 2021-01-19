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
import torch
import numpy as np
import random
import shutil
import os, cv2, dill
from matplotlib import pyplot as plt
from skimage.transform import resize
import ultra.utils.geometry as geometry
from scipy.spatial.distance import euclidean
import math


def rotate2d_vector(vectors, angle):
    ae_cos = np.cos(angle)
    ae_sin = np.sin(angle)
    rot_matrix = np.array([[ae_cos, -ae_sin], [ae_sin, ae_cos]])
    vectors_rotated = np.inner(vectors, rot_matrix)
    return vectors_rotated


def plot_experiment_data(experiment_path, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(f"{experiment_path}/results.pkl", "rb") as handle:
        data = dill.load(handle)
        iterations = sorted(data.keys())
        plot_names = data[iterations[0]].keys()
        for plot_name in plot_names:
            plt.figure()
            values = []
            for itr in iterations:
                values.append(data[itr][plot_name])
            plt.plot(iterations, values)
            plt.savefig(f"{save_dir}/{plot_name}.png")
            plt.close()


def seed_all_libraries(seed, max_rd=100000):
    np.random.seed(seed)
    rd_seed, np_seed, torch_seed = (
        np.random.randint(1, max_rd),
        np.random.randint(1, max_rd),
        np.random.randint(1, max_rd),
    )
    random.seed(rd_seed)
    np.random.seed(np_seed)
    torch.cuda.manual_seed_all(torch_seed)


def to_3d_action(action):
    throttle, steering = action
    brake = 0.0
    if throttle < 0.0:
        brake = abs(throttle)
        throttle = 0.0
    return np.asarray([throttle, brake, steering])


def to_2d_action(action):
    throttle, brake, steering = action
    if throttle != 0:
        accel = throttle
    else:
        accel = -brake if brake > 0 else 0.0
    return np.asarray([accel, steering])


def merge_discrete_action_spaces(*discrete_action_spaces):
    r = np.meshgrid(*discrete_action_spaces)
    r = np.column_stack([each.flatten() for each in r])
    index_to_action = [e.tolist() for e in r]
    action_to_index = {
        str(k): v
        for k, v in zip(index_to_action, np.arange(len(index_to_action)).astype(np.int))
    }
    return index_to_action, action_to_index


def deg2rad(x):
    return x / 180.0 * np.pi


def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            if os.path.exists(d):
                shutil.rmtree(d)
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)


def copy_source_code(srcs, dst):
    for s in srcs:
        d = os.path.join(dst, os.path.basename(s))
        if os.path.isdir(s):
            os.makedirs(d, exist_ok=True)
        (copytree if os.path.isdir(s) else shutil.copy)(s, d)


def resize_im(im, size):
    im = (resize(im, size, anti_aliasing=True, mode="constant") * 255.0).astype(
        np.uint8
    )
    return im


def normalize_im(x):
    # normalize from 0->255.0 to [-1, 1]
    x = x / 255.0
    x = 2.0 * x - 1
    return x


def get_box(vehicle, safety_dist):

    heading = vehicle.heading
    if heading > np.pi:
        heading = heading - 2 * np.pi

    del_dist = safety_dist / 2.0
    x_dist = -del_dist * np.sin(heading)
    y_dist = del_dist * np.cos(heading)

    return geometry.Box(
        width=vehicle.bounding_box.width,
        height=vehicle.bounding_box.length + safety_dist,
        length=vehicle.bounding_box.height,  # height of vehicle
        centerX=0.0,
        centerY=0.0,
        centerZ=0.0,
        yaw=0.0,
        roll=0.0,
        pitch=heading,  # regular yaw
        translationX=vehicle.position[0] + x_dist,
        translationY=vehicle.position[1] + y_dist,
        translationZ=vehicle.position[2],
    )


def ego_social_safety(
    agent_obs,
    d_min_ego=0.01,
    t_c_ego=0.5,
    d_min_social=0.01,
    t_c_social=0.5,
    ignore_vehicle_behind=False,
):
    # coordinate of this thing:
    # position: [x, y, z]
    # x increase when go from left to right
    # y increase when go from bottom to top
    # angle increase when vehicle move counterclockwise direction with 0 radian = bottom to top direction
    # so this is just standard coordinate shifted by pi/2 radian counterclockwise
    # add pi/2 radian to every heading to obtain the heading in standard coordinate

    def get_relative_position_vector_angle(v1, v2):
        x = v2.position[0] - v1.position[0]
        y = v2.position[1] - v1.position[1]
        # angle = clip_angle_to_pi(np.arctan2(y, x) + np.pi / 2)
        angle = clip_angle_to_pi(np.arctan2(x, y))
        return angle

    def is_behind(ego_heading_relative_diff, threshold=np.pi / 8):
        return abs(ego_heading_relative_diff) < threshold

    def get_vehicles_not_behind(ego, socials):
        ego_angle = ego.heading
        relative_position_vector_angles = [
            -get_relative_position_vector_angle(e, ego) for e in socials
        ]
        ego_heading_relative_diffs = [
            clip_angle_to_pi(ego_angle - e) for e in relative_position_vector_angles
        ]
        idxs = [
            e
            for e in range(len(socials))
            if not is_behind(ego_heading_relative_diffs[e])
        ]
        vehicles_not_behind = [socials[e] for e in idxs]
        return vehicles_not_behind

    # for debugging, you can use visualize_social_safety() in scenarios/visualization.py

    neighborhood_vehicle_states = agent_obs.neighborhood_vehicle_states
    ego_vehicle_state = agent_obs.ego_vehicle_state
    if ignore_vehicle_behind:
        neighborhood_vehicle_states = get_vehicles_not_behind(
            ego_vehicle_state, neighborhood_vehicle_states
        )

    vehicles_bounding_boxes = []
    vehicles_bounding_boxes_safety = []

    safety_dist_ego = d_min_ego + ego_vehicle_state.speed * t_c_ego

    ego_bounding_box_safety = get_box(ego_vehicle_state, safety_dist=safety_dist_ego)
    ego_bounding_box = get_box(ego_vehicle_state, safety_dist=0.0)

    for vehicle in neighborhood_vehicle_states:
        safety_dist_social = d_min_social + vehicle.speed * t_c_social
        vehicles_bounding_boxes.append(get_box(vehicle, safety_dist=0.0))
        vehicles_bounding_boxes_safety.append(
            get_box(vehicle, safety_dist=safety_dist_social)
        )

    ego_num_violations = 0
    social_num_violations = 0
    for vehicle, vehicle_safety in zip(
        vehicles_bounding_boxes, vehicles_bounding_boxes_safety
    ):
        if ego_bounding_box_safety.intersects(vehicle):
            ego_num_violations += 1
        if ego_bounding_box.intersects(vehicle_safety):
            social_num_violations += 1

    return ego_num_violations, social_num_violations


def get_closest_waypoint(ego_position, ego_heading, num_lookahead, goal_path):

    closest_wp = min(goal_path, key=lambda wp: wp.dist_to(ego_position))

    min_dist = float("inf")
    min_dist_idx = -1
    for i, wp in enumerate(goal_path):
        if wp.dist_to(ego_position) < min_dist:
            min_dist = wp.dist_to(ego_position)
            min_dist_idx = i
            closest_wp = wp

    waypoints_lookahead = [
        get_relative_pos(wp, ego_position)
        for wp in goal_path[
            min_dist_idx : min(min_dist_idx + num_lookahead, len(goal_path))
        ]
    ]
    if len(waypoints_lookahead) > 0:
        while len(waypoints_lookahead) < num_lookahead:
            waypoints_lookahead.append(waypoints_lookahead[-1])
    else:
        waypoints_lookahead = [
            get_relative_pos(closest_wp.pos, ego_position) for i in range(num_lookahead)
        ]

    waypoints_lookahead = rotate2d_vector(waypoints_lookahead, -ego_heading)

    return closest_wp, waypoints_lookahead


def get_relative_pos(waypoint, ego_pos):
    return [waypoint.pos[0] - ego_pos[0], waypoint.pos[1] - ego_pos[1]]


def get_dist_to_center(env_observation, closest_waypoint):
    ego = env_observation.ego_vehicle_state
    signed_dist_from_center = closest_waypoint.signed_lateral_error(ego.position)
    lane_hwidth = closest_waypoint.lane_width * 0.5
    norm_dist_from_center = signed_dist_from_center / lane_hwidth
    return norm_dist_from_center


def get_closest_point_index(pts_arr, pts):
    distance = [euclidean(each, pts) for each in pts_arr]
    return np.argmin(distance)


def get_path_to_goal(goal, paths, start):
    goal_pos = goal.position
    start_pos = start.position
    path_start_pts = [each[0].pos for each in paths]

    best_path_ind = get_closest_point_index(path_start_pts, start_pos)
    path = paths[best_path_ind]

    middle_point = path[int(len(path) / 2)]
    goal_lane_id = middle_point.lane_id
    goal_lane_index = middle_point.lane_index

    path_pts = [each.pos for each in path]
    end_path_ind = get_closest_point_index(path_pts, goal_pos)

    return path


def compute_sum_aux_losses(aux_losses):
    return sum([e["value"] * e["weight"] for e in aux_losses.values()])


def clip_angle_to_pi(angle):
    while angle < -np.pi:
        angle += np.pi * 2
    while angle > np.pi:
        angle -= np.pi * 2
    return angle


def compute_grad(x, prev_x, dt):
    grad = (x - prev_x) / dt if prev_x is not None else x / dt
    return grad
