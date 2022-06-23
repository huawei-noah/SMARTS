from typing import Callable, Dict

import numpy as np
from custom_dict import CustomDict

from smarts.core.sensors import Observation
from smarts.env.custom_observations import lane_ttc


class Score:
    def __init__(self):
        self._results = {}


class Metric:
    def __init__(self, agent_names):
        costs = {
            "collisions": 0,
            "on_shoulder": 0,
            "reached_goal": 0,
            "lane_center_offset": 0,
            "dist_to_obstacles": 0,
            "steering_rate": 0,
            "jerk": 0,
        }
        self._results = {name: CustomDict(**costs) for name in agent_names}
        self._steering_rate = _steering_rate()

    def store(self, infos):
        for agent_name, agent_info in infos.items():
            agent_obs = agent_info["env_obs"]
            costs = dict(
                **(_collisions(agent_obs)),
                **(_on_shoulder(agent_obs)),
                **(_reached_goal(agent_obs)),
                **(self._steering_rate(agent_obs)),
                **(_jerk(agent_obs)),
                **(_lane_ttc(agent_obs)),
            )
            self._results[agent_name] += costs

    def compute(self):
        self._results


def _collisions(obs: Observation) -> Dict[str, int]:
    return {"collisions": len(obs.events.collisions)}


def _on_shoulder(obs: Observation) -> Dict[str, int]:
    return {"on_shoulder": int(obs.events.on_shoulder)}


def _reached_goal(obs: Observation) -> Dict[str, int]:
    return {"reached_goal": int(obs.events.reached_goal)}


def _lane_ttc(obs: Observation) -> Dict[str, float]:
    val = lane_ttc(obs)

    # J_LC : Lane center offset
    jlc = np.float32((val["distance_from_center"][0]) ** 2)

    # J_D : Distance to obstacles in front
    w_dist = 0.1
    dist_to_obstacles = np.array(val["ego_lane_dist"], dtype=np.float32)
    jd = np.amax(np.exp(-w_dist * dist_to_obstacles))

    return {"lane_center_offset": jlc, "dist_to_obstacles": jd}


def _steering_rate() -> Callable[[Observation], Dict[str, float]]:
    prev_steering = 0

    def func(obs: Observation) -> Dict[str, float]:
        nonlocal prev_steering
        velocity = (obs.ego_vehicle_state.steering - prev_steering) / 0.1
        prev_steering = obs.ego_vehicle_state.steering
        return {"steering_rate": velocity ** 2}

    return func


def _jerk(obs: Observation) -> Dict[str, float]:
    w_jerk = [0.7, 0.3]

    l_jerk_2 = (obs.ego_vehicle_state.linear_jerk) ** 2
    a_jerk_2 = (obs.ego_vehicle_state.angular_jerk) ** 2
    jerk = np.dot(w_jerk, [l_jerk_2, a_jerk_2])

    return {"jerk": jerk}
