from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np

from smarts.core.sensors import Observation


@dataclass
class Costs:
    collisions: int = 0
    dist_to_goal: float = 0
    dist_to_obstacles: float = 0
    jerk_angular: float = 0
    jerk_linear: float = 0
    lane_center_offset: float = 0
    off_road: int = 0
    speed_limit: float = 0
    wrong_way: int = 0


COST_FUNCS = {
    "collisions": lambda: _collisions,
    "dist_to_goal": lambda: _dist_to_goal,
    "dist_to_obstacles": lambda: _dist_to_obstacles(),
    "jerk_angular": lambda: _jerk_angular(),
    "jerk_linear": lambda: _jerk_linear(),
    "lane_center_offset": lambda: _lane_center_offset(),
    "off_road": lambda: _off_road,
    "speed_limit": lambda: _speed_limit(),
    "wrong_way": lambda: _wrong_way(),
}


def _collisions(obs: Observation) -> Dict[str, int]:
    return {"collisions": len(obs.events.collisions)}


def _dist_to_goal(obs: Observation) -> Dict[str, float]:
    mission_goal = obs.ego_vehicle_state.mission.goal
    if hasattr(mission_goal, "position"):
        rel = obs.ego_vehicle_state.position[:2] - mission_goal.position[:2]
        dist = sum(abs(rel))
    else:
        dist = 0

    return {"dist_to_goal": dist}


def _dist_to_obstacles() -> Callable[[Observation], Dict[str, float]]:
    ave = 0
    step = 0
    rel_angle_th = np.pi * 40 / 180
    rel_heading_th = np.pi * 179 / 180
    w_dist = 0.05

    def func(obs: Observation) -> Dict[str, float]:
        nonlocal ave, step, rel_angle_th, rel_heading_th, w_dist

        # Ego's position and heading with respect to the map's coordinate system.
        # Note: All angles returned by smarts is with respect to the map's coordinate system.
        #       On the map, angle is zero at positive y axis, and increases anti-clockwise.
        ego = obs.ego_vehicle_state
        ego_heading = (ego.heading + np.pi) % (2 * np.pi) - np.pi
        ego_pos = ego.position

        # Set obstacle distance threshold using 3-second rule
        obstacle_dist_th = ego.speed * 3
        if obstacle_dist_th == 0:
            return {"dist_to_obstacles": 0}

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
            return {"dist_to_obstacles": 0}

        # Filter neighbors within ego's visual field.
        obstacles = []
        for nghb_state in nghbs_state:
            # Neighbors's angle with respect to the ego's position.
            # Note: In np.angle(), angle is zero at positive x axis, and increases anti-clockwise.
            #       Hence, map_angle = np.angle() - π/2
            rel_pos = np.array(nghb_state[0].position) - ego_pos
            obstacle_angle = np.angle(rel_pos[0] + 1j * rel_pos[1]) - np.pi / 2
            obstacle_angle = (obstacle_angle + np.pi) % (2 * np.pi) - np.pi
            # Relative angle is the angle correction required by ego agent to face the obstacle.
            rel_angle = obstacle_angle - ego_heading
            rel_angle = (rel_angle + np.pi) % (2 * np.pi) - np.pi
            if abs(rel_angle) <= rel_angle_th:
                obstacles.append(nghb_state)
        nghbs_state = obstacles
        if len(nghbs_state) == 0:
            return {"dist_to_obstacles": 0}

        # Filter neighbors by their relative heading to that of ego's heading.
        nghbs_state = [
            nghb_state
            for nghb_state in nghbs_state
            if abs(nghb_state[0].heading.relative_to(ego.heading)) <= rel_heading_th
        ]
        if len(nghbs_state) == 0:
            return {"dist_to_obstacles": 0}

        # J_D : Distance to obstacles cost
        di = [nghb_state[1] for nghb_state in nghbs_state]
        di = np.array(di)
        j_d = np.amax(np.exp(-w_dist * di))

        ave, step = _running_ave(prev_ave=ave, prev_step=step, new_val=j_d)
        return {"dist_to_obstacles": ave}

    return func


def _jerk_angular() -> Callable[[Observation], Dict[str, float]]:
    ave = 0
    step = 0

    def func(obs: Observation) -> Dict[str, float]:
        nonlocal ave, step

        ja_squared = np.sum(np.square(obs.ego_vehicle_state.angular_jerk))
        ave, step = _running_ave(prev_ave=ave, prev_step=step, new_val=ja_squared)
        return {"jerk_angular": ave}

    return func


def _jerk_linear() -> Callable[[Observation], Dict[str, float]]:
    ave = 0
    step = 0

    def func(obs: Observation) -> Dict[str, float]:
        nonlocal ave, step

        jl_squared = np.sum(np.square(obs.ego_vehicle_state.linear_jerk))
        ave, step = _running_ave(prev_ave=ave, prev_step=step, new_val=jl_squared)
        return {"jerk_linear": ave}

    return func


def _lane_center_offset() -> Callable[[Observation], Dict[str, float]]:
    ave = 0
    step = 0

    def func(obs: Observation) -> Dict[str, float]:
        nonlocal ave, step

        # Nearest waypoints
        ego = obs.ego_vehicle_state
        waypoint_paths = obs.waypoint_paths
        wps = [path[0] for path in waypoint_paths]

        # Distance of vehicle from center of lane
        closest_wp = min(wps, key=lambda wp: wp.dist_to(ego.position))
        signed_dist_from_center = closest_wp.signed_lateral_error(ego.position)
        lane_hwidth = closest_wp.lane_width * 0.5
        norm_dist_from_center = signed_dist_from_center / lane_hwidth

        # J_LC : Lane center offset
        j_lc = norm_dist_from_center ** 2

        ave, step = _running_ave(prev_ave=ave, prev_step=step, new_val=j_lc)
        return {"lane_center_offset": ave}

    return func


def _off_road(obs: Observation) -> Dict[str, int]:
    return {"off_road": obs.events.off_road}


def _speed_limit() -> Callable[[Observation], Dict[str, float]]:
    ave = 0
    step = 0

    def func(obs: Observation) -> Dict[str, float]:
        nonlocal ave, step

        # Nearest waypoints.
        ego = obs.ego_vehicle_state
        waypoint_paths = obs.waypoint_paths
        wps = [path[0] for path in waypoint_paths]

        # Speed limit.
        closest_wp = min(wps, key=lambda wp: wp.dist_to(ego.position))
        speed_limit = closest_wp.speed_limit

        # Excess speed beyond speed limit.
        overspeed = ego.speed - speed_limit if ego.speed > speed_limit else 0
        j_v = overspeed ** 2

        ave, step = _running_ave(prev_ave=ave, prev_step=step, new_val=j_v)
        return {"speed_limit": ave}

    return func


def _wrong_way() -> Callable[[Observation], Dict[str, float]]:
    ave = 0
    step = 0

    def func(obs: Observation) -> Dict[str, float]:
        nonlocal ave, step
        wrong_way = 0
        if obs.events.wrong_way:
            wrong_way = 1

        ave, step = _running_ave(prev_ave=ave, prev_step=step, new_val=wrong_way)
        return {"wrong_way": ave}

    return func


def _running_ave(prev_ave: float, prev_step: int, new_val: float) -> Tuple[float, int]:
    new_step = prev_step + 1
    new_ave = prev_ave + (new_val - prev_ave) / new_step
    return new_ave, new_step
