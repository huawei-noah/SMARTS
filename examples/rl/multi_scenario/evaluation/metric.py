from typing import Dict

import numpy as np
from custom_dict import CustomDict

from smarts.core.sensors import Observation


class Score:
    def __init__(self):
        self._results = {}


class Metric:
    def __init__(self, agent_names):
        costs = {
            "collisions": 0,
            "dist_to_obstacles": 0,
            "jerk": 0,
            "lane_center_offset": 0,
            "on_shoulder": 0,
            "reached_goal": 0,
            "steering_rate": 0,
            "yaw_rate": 0,
        }
        self._runningresults = {name: CustomDict(**costs) for name in agent_names}
        self._steering_rate = _SteeringRate()
        self._steering_rate = _Overtake()
        self._steering_rate = _Steps()

    def store(self, infos):
        for agent_name, agent_info in infos.items():
            agent_obs = agent_info["env_obs"]
            costs = dict(
                **(_collisions(agent_obs)),
                **(_on_shoulder(agent_obs)),
                **(_reached_goal(agent_obs)),
                **(_lane_center_offset(agent_obs)),
                **(self._steering_rate(agent_obs, agent_name)),
                **(_jerk(agent_obs)),
                **(_yaw_rate(agent_obs)),
                **(_distance_to_obstacles(agent_obs)),
            )
            self._results[agent_name] += costs

    def reinit(self):
        self._steering_rate.reinit()

    def compute(self):
        pass


def _collisions(obs: Observation) -> Dict[str, int]:
    return {"collisions": len(obs.events.collisions)}


def _distance_to_obstacles(obs: Observation) -> Dict[str, float]:
    obstacle_dist_th = 50
    obstacle_angle_th = np.pi * 30 / 180
    w_dist = 0.1

    # Ego's position and angle with respect to the map's axes.
    # Note: All angles returned by smarts is with respect to the map axes.
    #       Angle is zero at positive y axis, and increases anti-clockwise, on the map.
    ego_angle = (obs["ego"]["heading"] + np.pi) % (2 * np.pi) - np.pi
    ego_pos = obs["ego"]["position"]

    # Filter neighbors by distance
    nghbs = obs.neighborhood_vehicle_states
    nghbs = [
        (nghb.id, nghb.position, np.linalg.norm(ego_pos - nghb.position))
        for nghb in nghbs
    ]
    nghbs = filter(lambda x: x[2] <= obstacle_dist_th, nghbs)
    if len(nghbs) == 0:
        return 0

    # Filter neighbors by angle
    obstacles = []
    for id, pos, dist in nghbs:
        # Neighbors's angle with respect to the map's axes.
        # Note: Angle is zero at positive x axis, and increases anti-clockwise, in np.angle().
        #       Hence, map_angle = np.angle() - π/2
        obstacle_angle = np.angle(pos[0] + 1j * pos[1]) - np.pi / 2
        obstacle_angle = (obstacle_angle + np.pi) % (2 * np.pi) - np.pi
        # Obstacle heading is the angle correction required by ego agent to face the obstacle.
        obstacle_heading = obstacle_angle - ego_angle
        obstacle_heading = (obstacle_heading + np.pi) % (2 * np.pi) - np.pi
        if abs(obstacle_heading) <= obstacle_angle_th:
            obstacles.append((id, pos, dist, obstacle_heading))

    # J_D : Distance to obstacles cost
    _, _, di, _ = zip(**obstacles)
    for obstacle in obstacles:
        print(f"Obstacle: {obstacle[0]}, {obstacle[1]}, {obstacle[2]}, {obstacle[3]}.")
    j_d = np.amax(np.exp(-w_dist * di))

    return j_d


def _jerk(obs: Observation) -> Dict[str, float]:
    w_jerk = [0.7, 0.3]

    lj_squared = np.sum(np.square(obs.ego_vehicle_state.linear_jerk))
    aj_squared = np.sum(np.square(obs.ego_vehicle_state.angular_jerk))

    jerk = np.dot(w_jerk, [lj_squared, aj_squared])

    return {"jerk": jerk}


def _lane_center_offset(obs: Observation) -> Dict[str, float]:
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
    jlc = np.float32(norm_dist_from_center ** 2)

    return {"lane_center_offset": jlc}


class _Overtake():
    def __init__(self, agent_names):
        self._lane = {name: 0 for name in agent_names}

    def reinit(self):
        self._lane = {name: 0 for name in self._lane.keys()}

    def __call__(self, obs: Observation, agent_name: str):
        lane_index = obs.ego_vehicle_state.lane_index
        self._lane[agent_name].append(lane_index)

    def check(agent_name: str):
        overtake = 0

        return {"overtake": overtake}


def _on_shoulder(obs: Observation) -> Dict[str, int]:
    return {"on_shoulder": int(obs.events.on_shoulder)}


def _reached_goal(obs: Observation) -> Dict[str, int]:
    return {"reached_goal": int(obs.events.reached_goal)}


class _SteeringRate():
    def __init__(self, agent_names):
        self._prev_steering = {name: 0 for name in agent_names}

    def reinit(self):
        self._prev_steering = {name: 0 for name in self._prev_steering.keys()}

    def __call__(self, obs: Observation, agent_name: str) -> Dict[str, float]:
        velocity = (
            obs.ego_vehicle_state.steering - self._prev_steering[agent_name]
        ) / 0.1
        self._prev_steering[agent_name] = obs.ego_vehicle_state.steering
        return {"steering_rate": velocity ** 2}


class _Steps():
    def __init__(self, agent_names):
        prev_mean_steps = 0
        prev_reached_goal = 0

    def __call_(self, obs:Observation, agent_name: str):
        pass

    # def func(obs: Observation) -> Dict[str, float]:
    #     nonlocal prev_mean_steps

    #     # Only if you reached goal
    #     obs.events.reached_goal

    #     new_mean_steps = prev_mean_steps + ()/
    #     prev_steering = obs.ego_vehicle_state.steering

    #     return {"mean_time": velocity ** 2}


def _yaw_rate(obs: Observation) -> Dict[str, float]:
    yr_squared = obs.ego_vehicle_state.yaw_rate ** 2
    return {"yaw_rate": yr_squared}
