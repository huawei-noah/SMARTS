from custom_dict import CustomDict
from evaluation.metric import COST_FUNCS


class Score:
    def __init__(self):
        self._results = {
            "completion": 0,
            "humanness": 0,
            "rules": 0,
            "time": 0,
        }
        self._costs = CustomDict(**{key: 0 for key in COST_FUNCS.keys()})

    def add(self, metric):
        for agent_costs in metric["costs"].values():
            self._costs += agent_costs
        self._episodes += metric["episodes"]
        self._incomplete += metric["incomplete"]
        self._steps += metric["steps"]
        self._adjusted_steps += metric["adjusted_steps"]
        self._goals_unachieved += metric["goals_unachieved"]

    def compute(self):
        self._results["completion"] = _completion(
            self._incomplete, self._goals_unachieved, self._episodes
        )
        self._results["humanness"] = _humanness(self._costs, self._steps)
        self._results["rules"] = _rules(self._costs, self._episodes)
        self._results["time"] = _time(self._adjusted_steps, self._episodes)

        return self._results


def _completion(incomplete, goals_unachieved, episodes):
    w_ic = 0.6
    w_gua = 0.4

    return (w_ic * incomplete + w_gua * goals_unachieved) / episodes


# Need to modify to take average of average jerkiness
def _humanness(costs, steps):
    w_d = 0.2
    w_j = 0.2
    w_lc = 0.2
    w_sr = 0.15
    w_vo = 0.15
    w_yr = 0.1

    return (
        w_d * costs["dist_to_obstacles"]
        + w_j * costs["jerk"]
        + w_lc * costs["lane_center_offset"]
        + w_sr * costs["steering_rate"]
        + w_vo * costs["velocity_offset"]
        + w_yr * costs["yaw_rate"]
    ) / steps


def _rules(costs, episodes):
    w_c = 0.2
    w_ord = 0.2
    w_ort = 0.2
    w_os = 0.2
    w_ww = 0.2

    return (
        w_c * costs["collisions"]
        + w_ord * costs["off_road"]
        + w_ort * costs["off_route"]
        + w_os * costs["on_shoulder"]
        + w_ww * costs["wrong_way"]
    ) / episodes


def _time(adjusted_steps, episodes):
    return adjusted_steps / episodes
