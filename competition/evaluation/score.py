from dataclasses import asdict
from typing import Dict

from costs import Costs
from metric import Counts


class Score:
    def __init__(self):
        self._results = {
            "completion": 0,
            "humanness": 0,
            "rules": 0,
            "time": 0,
        }
        self._counts = Counts()
        self._costs = Costs()

    @property
    def keys(self):
        return list(self._results.keys())

    def add(self, counts: Counts, costs: Costs):
        for count_name, count_val in asdict(counts).items():
            new_val = getattr(self._counts, count_name) + count_val
            setattr(self._counts, count_name, new_val)

        for cost_name, cost_val in asdict(costs).items():
            new_val = getattr(self._costs, cost_name) + cost_val
            setattr(self._costs, cost_name, new_val)

    def compute(self) -> Dict[str, float]:
        self._results["completion"] = _completion(counts=self._counts)
        self._results["humanness"] = _humanness(counts=self._counts, costs=self._costs)
        self._results["rules"] = _rules(counts=self._counts, costs=self._costs)
        self._results["time"] = _time(counts=self._counts)

        return self._results


def _completion(counts: Counts) -> float:
    w_cr = 0.6
    w_pt = 0.4

    return (w_cr * counts.crashes + w_pt * counts.partial) / counts.episodes


def _humanness(counts: Counts, costs: Costs) -> float:
    w_d = 0.25
    w_j = 0.25
    w_lc = 0.2
    w_vo = 0.15
    w_yr = 0.15

    return (
        w_d * costs.dist_to_obstacles
        + w_j * costs.jerk
        + w_lc * costs.lane_center_offset
        + w_vo * costs.velocity_offset
        + w_yr * costs.yaw_rate
    ) / counts.episodes_adjusted


def _rules(counts: Counts, costs: Costs) -> float:
    w_c = 0.3
    w_ord = 0.175
    w_ort = 0.1
    w_os = 0.175
    w_ww = 0.25

    return (
        w_c * costs.collisions
        + w_ord * costs.off_road
        + w_ort * costs.off_route
        + w_os * costs.on_shoulder
        + w_ww * costs.wrong_way
    ) / counts.episodes_adjusted


def _time(counts: Counts) -> float:
    return counts.steps_adjusted / counts.episodes
