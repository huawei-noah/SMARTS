# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from dataclasses import dataclass
from typing import Any, Callable, Dict

import numpy as np

from smarts.core.coordinates import Heading, Point
from smarts.core.observations import Observation
from smarts.core.road_map import RoadMap
from smarts.core.utils.math import running_mean


@dataclass(frozen=True)
class Costs:
    """Performance cost values."""

    collisions: int = 0
    dist_to_obstacles: float = 0
    jerk_angular: float = 0
    jerk_linear: float = 0
    lane_center_offset: float = 0
    off_road: int = 0
    speed_limit: float = 0
    wrong_way: float = 0


def _collisions(road_map: RoadMap, obs: Observation) -> Costs:
    return Costs(collisions=len(obs.events.collisions))


def _dist_to_obstacles() -> Callable[[RoadMap, Observation], Costs]:
    mean = 0
    step = 0
    rel_angle_th = np.pi * 40 / 180
    rel_heading_th = np.pi * 179 / 180
    w_dist = 0.05

    def func(road_map: RoadMap, obs: Observation) -> Costs:
        nonlocal mean, step, rel_angle_th, rel_heading_th, w_dist

        # Ego's position and heading with respect to the map's coordinate system.
        # Note: All angles returned by smarts is with respect to the map's coordinate system.
        #       On the map, angle is zero at positive y axis, and increases anti-clockwise.
        ego = obs.ego_vehicle_state
        ego_heading = (ego.heading + np.pi) % (2 * np.pi) - np.pi
        ego_pos = ego.position

        # Set obstacle distance threshold using 3-second rule
        obstacle_dist_th = ego.speed * 3
        if obstacle_dist_th == 0:
            return Costs(dist_to_obstacles=0)

        # Get neighbors.
        nghbs = obs.neighborhood_vehicle_states

        # Filter neighbors by distance.
        nghbs_state = [
            (nghb, np.linalg.norm(nghb.position - ego_pos)) for nghb in nghbs
        ]
        nghbs_state = [
            nghb_state
            for nghb_state in nghbs_state
            if nghb_state[1] <= obstacle_dist_th
        ]
        if len(nghbs_state) == 0:
            return Costs(dist_to_obstacles=0)

        # Filter neighbors within ego's visual field.
        obstacles = []
        for nghb_state in nghbs_state:
            # Neighbors's angle with respect to the ego's position.
            # Note: In np.angle(), angle is zero at positive x axis, and increases anti-clockwise.
            #       Hence, map_angle = np.angle() - π/2
            rel_pos = np.array(nghb_state[0].position) - ego_pos
            obstacle_angle = np.angle(rel_pos[0] + 1j * rel_pos[1]) - np.pi / 2
            obstacle_angle = (obstacle_angle + np.pi) % (2 * np.pi) - np.pi
            # Relative angle is the angle rotation required by ego agent to face the obstacle.
            rel_angle = obstacle_angle - ego_heading
            rel_angle = (rel_angle + np.pi) % (2 * np.pi) - np.pi
            if abs(rel_angle) <= rel_angle_th:
                obstacles.append(nghb_state)
        nghbs_state = obstacles
        if len(nghbs_state) == 0:
            return Costs(dist_to_obstacles=0)

        # Filter neighbors by their relative heading to that of ego's heading.
        nghbs_state = [
            nghb_state
            for nghb_state in nghbs_state
            if abs(nghb_state[0].heading.relative_to(ego.heading)) <= rel_heading_th
        ]
        if len(nghbs_state) == 0:
            return Costs(dist_to_obstacles=0)

        # J_D : Distance to obstacles cost
        di = np.array([nghb_state[1] for nghb_state in nghbs_state])
        j_d = np.amax(np.exp(-w_dist * di))

        mean, step = running_mean(prev_mean=mean, prev_step=step, new_val=j_d)
        return Costs(dist_to_obstacles=mean)

    return func


def _jerk_angular() -> Callable[[RoadMap, Observation], Costs]:
    mean = 0
    step = 0

    # TODO: The output of this cost function should be normalised and bounded to [0,1].
    def func(road_map: RoadMap, obs: Observation) -> Costs:
        nonlocal mean, step

        j_a = np.linalg.norm(obs.ego_vehicle_state.angular_jerk)
        mean, step = running_mean(prev_mean=mean, prev_step=step, new_val=j_a)
        return Costs(jerk_angular=mean)

    return func


def _jerk_linear() -> Callable[[RoadMap, Observation], Costs]:
    mean = 0
    step = 0
    jerk_linear_max = np.linalg.norm(np.array([0.9, 0.9, 0]))  # Units: m/s^3
    """
    Maximum comfortable linear jerk as presented in:

    Bae, Il and et. al., "Self-Driving like a Human driver instead of a
    Robocar: Personalized comfortable driving experience for autonomous vehicles", 
    Machine Learning for Autonomous Driving Workshop at the 33rd Conference on 
    Neural Information Processing Systems, NeurIPS 2019, Vancouver, Canada.
    """

    def func(road_map: RoadMap, obs: Observation) -> Costs:
        nonlocal mean, step, jerk_linear_max

        jerk_linear = np.linalg.norm(obs.ego_vehicle_state.linear_jerk)
        j_l = min(jerk_linear / jerk_linear_max, 1)
        mean, step = running_mean(prev_mean=mean, prev_step=step, new_val=j_l)
        return Costs(jerk_linear=mean)

    return func


def _lane_center_offset() -> Callable[[RoadMap, Observation], Costs]:
    mean = 0
    step = 0

    def func(road_map: RoadMap, obs: Observation) -> Costs:
        nonlocal mean, step

        if obs.events.off_road:
            # When vehicle is off road, the lane_center_offset cost
            # (i.e., j_lc) is set as zero.
            j_lc = 0
        else:
            # Vehicle's offset along the lane
            ego_pos = obs.ego_vehicle_state.position
            ego_lane = road_map.lane_by_id(obs.ego_vehicle_state.lane_id)
            reflinepoint = ego_lane.to_lane_coord(world_point=Point(*ego_pos))

            # Half width of lane
            lane_width, _ = ego_lane.width_at_offset(reflinepoint.s)
            lane_hwidth = lane_width * 0.5

            # Normalized vehicle's displacement from lane center
            # reflinepoint.t = signed distance from lane center
            norm_dist_from_center = reflinepoint.t / lane_hwidth

            # J_LC : Lane center offset
            j_lc = norm_dist_from_center**2

        mean, step = running_mean(prev_mean=mean, prev_step=step, new_val=j_lc)
        return Costs(lane_center_offset=mean)

    return func


def _off_road(road_map: RoadMap, obs: Observation) -> Costs:
    return Costs(off_road=obs.events.off_road)


def _speed_limit() -> Callable[[RoadMap, Observation], Costs]:
    mean = 0
    step = 0

    def func(road_map: RoadMap, obs: Observation) -> Costs:
        nonlocal mean, step

        if obs.events.off_road:
            # When vehicle is off road, the speed_limit cost (i.e., j_v) is
            # set as zero.
            j_v = 0
        else:
            # Nearest lane's speed limit.
            ego_speed = obs.ego_vehicle_state.speed
            ego_lane = road_map.lane_by_id(obs.ego_vehicle_state.lane_id)
            speed_limit = ego_lane.speed_limit
            assert speed_limit > 0, (
                "Expected lane speed limit to be a positive "
                f"float, but got speed_limit: {speed_limit}."
            )

            # Excess speed beyond speed limit.
            overspeed = ego_speed - speed_limit if ego_speed > speed_limit else 0
            overspeed_norm = min(overspeed / (0.5 * speed_limit), 1)
            j_v = overspeed_norm**2

        mean, step = running_mean(prev_mean=mean, prev_step=step, new_val=j_v)
        return Costs(speed_limit=mean)

    return func


def _wrong_way() -> Callable[[RoadMap, Observation], Costs]:
    mean = 0
    step = 0

    def func(road_map: RoadMap, obs: Observation) -> Costs:
        nonlocal mean, step
        j_wrong_way = 0
        if obs.events.wrong_way:
            j_wrong_way = 1

        mean, step = running_mean(prev_mean=mean, prev_step=step, new_val=j_wrong_way)
        return Costs(wrong_way=mean)

    return func


@dataclass(frozen=True)
class CostFuncs:
    """Functions to compute performance costs. Each cost function computes the
    running mean cost over number of time steps, for a given scenario."""

    collisions: Callable[[RoadMap, Observation], Costs] = _collisions
    dist_to_obstacles: Callable[[RoadMap, Observation], Costs] = _dist_to_obstacles()
    # jerk_angular: Callable[[RoadMap, Observation], Costs] = _jerk_angular() # Currently not used.
    jerk_linear: Callable[[RoadMap, Observation], Costs] = _jerk_linear()
    lane_center_offset: Callable[[RoadMap, Observation], Costs] = _lane_center_offset()
    off_road: Callable[[RoadMap, Observation], Costs] = _off_road
    speed_limit: Callable[[RoadMap, Observation], Costs] = _speed_limit()
    wrong_way: Callable[[RoadMap, Observation], Costs] = _wrong_way()
