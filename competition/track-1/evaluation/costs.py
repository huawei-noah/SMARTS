import re
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np

from smarts.core.sensors import Observation


@dataclass
class Costs:
    collisions: int = 0
    dist_to_obstacles: float = 0
    jerk: float = 0
    lane_center_offset: float = 0
    off_road: int = 0
    off_route: int = 0
    on_shoulder: int = 0
    steering_rate: float = 0
    velocity_offset: float = 0
    wrong_way: int = 0
    yaw_rate: float = 0


COST_FUNCS = {
    "collisions": lambda: _collisions,
    "dist_to_obstacles": lambda: _dist_to_obstacles(),
    "jerk": lambda: _jerk(),
    "lane_center_offset": lambda: _lane_center_offset(),
    "off_road": lambda: _off_road,
    "off_route": lambda: _off_route,
    "on_shoulder": lambda: _on_shoulder,
    "steering_rate": lambda: _steering_rate(),
    "velocity_offset": lambda: _velocity_offset(),
    "wrong_way": lambda: _wrong_way,
    "yaw_rate": lambda: _yaw_rate(),
}


def _collisions(obs: Observation) -> Dict[str, int]:
    return {"collisions": len(obs.events.collisions)}


def _dist_to_obstacles() -> Callable[[Observation], Dict[str, float]]:
    ave = 0
    step = 0
    regexp_jn = re.compile(r":.*J")
    obstacle_dist_th = 50
    obstacle_angle_th = np.pi * 40 / 180
    w_dist = 0.05

    def func(obs: Observation) -> Dict[str, float]:
        nonlocal ave, step, regexp_jn, obstacle_dist_th, obstacle_angle_th, w_dist

        # Ego's position and heading with respect to the map's axes.
        # Note: All angles returned by smarts is with respect to the map's axes.
        #       On the map, angle is zero at positive y axis, and increases anti-clockwise.
        ego = obs.ego_vehicle_state
        ego_heading = (ego.heading + np.pi) % (2 * np.pi) - np.pi
        ego_pos = ego.position
        lane_ids = [wp.lane_id for path in obs.waypoint_paths for wp in path]
        lane_ids = set(lane_ids)
        ego_road_ids = [id.split("_")[0] for id in lane_ids]
        ego_road_ids = set(ego_road_ids)

        # Get neighbors.
        nghbs = obs.neighborhood_vehicle_states

        # Filter neighbors by road id.
        nghbs = [
            nghb
            for nghb in nghbs
            if (
                # Match neighbor and ego road id.
                nghb.road_id == ego.road_id
                # Match neighbor road id to ':.*J' pattern.
                or regexp_jn.search(nghb.road_id)
                # Match neighbor road id to any road id in ego path.
                or nghb.road_id in ego_road_ids
            )
        ]

        if len(nghbs) == 0:
            return {"dist_to_obstacles": 0}

        # Filter neighbors by distance.
        nghbs = [
            (nghb.position, np.linalg.norm(nghb.position - ego_pos)) for nghb in nghbs
        ]
        nghbs = [nghb for nghb in nghbs if nghb[1] <= obstacle_dist_th]

        if len(nghbs) == 0:
            return {"dist_to_obstacles": 0}

        # Filter neighbors by angle.
        obstacles = []
        for pos, dist in nghbs:
            # Neighbors's angle with respect to the ego's position.
            # Note: In np.angle(), angle is zero at positive x axis, and increases anti-clockwise.
            #       Hence, map_angle = np.angle() - π/2
            rel_pos = pos - ego_pos
            obstacle_angle = np.angle(rel_pos[0] + 1j * rel_pos[1]) - np.pi / 2
            obstacle_angle = (obstacle_angle + np.pi) % (2 * np.pi) - np.pi
            # Obstacle heading is the angle correction required by ego agent to face the obstacle.
            obstacle_heading = obstacle_angle - ego_heading
            obstacle_heading = (obstacle_heading + np.pi) % (2 * np.pi) - np.pi
            if abs(obstacle_heading) <= obstacle_angle_th:
                obstacles.append((pos, dist, obstacle_heading))

        if len(obstacles) == 0:
            return {"dist_to_obstacles": 0}

        # J_D : Distance to obstacles cost
        _, di, _ = zip(*obstacles)
        di = np.array(di)
        j_d = np.amax(np.exp(-w_dist * di))

        ave, step = _running_ave(prev_ave=ave, prev_step=step, new_val=j_d)
        return {"dist_to_obstacles": ave}

    return func


def _jerk() -> Callable[[Observation], Dict[str, float]]:
    ave = 0
    step = 0
    w_jerk = [0.7, 0.3]

    def func(obs: Observation) -> Dict[str, float]:
        nonlocal ave, step, w_jerk

        lj_squared = np.sum(np.square(obs.ego_vehicle_state.linear_jerk))
        aj_squared = np.sum(np.square(obs.ego_vehicle_state.angular_jerk))
        j_j = np.dot(w_jerk, [lj_squared, aj_squared])

        ave, step = _running_ave(prev_ave=ave, prev_step=step, new_val=j_j)
        return {"jerk": ave}

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


def _off_route(obs: Observation) -> Dict[str, int]:
    return {"off_route": obs.events.off_route}


def _on_shoulder(obs: Observation) -> Dict[str, int]:
    return {"on_shoulder": int(obs.events.on_shoulder)}


def _steering_rate() -> Callable[[Observation], Dict[str, float]]:
    ave = 0
    step = 0
    prev_steering = 0

    def func(obs: Observation) -> Dict[str, float]:
        nonlocal ave, step, prev_steering
        steering_velocity = (obs.ego_vehicle_state.steering - prev_steering) / 0.1
        prev_steering = obs.ego_vehicle_state.steering
        j_sr = steering_velocity ** 2
        ave, step = _running_ave(prev_ave=ave, prev_step=step, new_val=j_sr)
        return {"steering_rate": ave}

    return func


def _velocity_offset() -> Callable[[Observation], Dict[str, float]]:
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
        overspeed = speed_limit - ego.speed if speed_limit > ego.speed else 0
        j_v = overspeed ** 2

        ave, step = _running_ave(prev_ave=ave, prev_step=step, new_val=j_v)
        return {"velocity_offset": ave}

    return func


def _wrong_way(obs: Observation) -> Dict[str, int]:
    return {"wrong_way": obs.events.wrong_way}


def _yaw_rate() -> Callable[[Observation], Dict[str, float]]:
    ave = 0
    step = 0

    def func(obs: Observation) -> Dict[str, float]:
        nonlocal ave, step
        j_y = obs.ego_vehicle_state.yaw_rate ** 2
        ave, step = _running_ave(prev_ave=ave, prev_step=step, new_val=j_y)
        return {"yaw_rate": ave}

    return func


def _running_ave(prev_ave: float, prev_step: int, new_val: float) -> Tuple[float, int]:
    new_step = prev_step + 1
    new_ave = prev_ave + (new_val - prev_ave) / new_step
    return new_ave, new_step
