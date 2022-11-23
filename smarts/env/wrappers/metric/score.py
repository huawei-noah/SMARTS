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
        self._results["time"] = _time(counts=self._counts, costs=self._costs)

        return self._results


def _completion(counts: Counts) -> float:
    return counts.crashes / counts.episodes


def _humanness(counts: Counts, costs: Costs) -> float:

    return (
        costs.dist_to_obstacles
        + costs.jerk_angular
        + costs.jerk_linear
        + costs.lane_center_offset
    ) / counts.episode_agents


def _rules(counts: Counts, costs: Costs) -> float:
    return (costs.speed_limit + costs.wrong_way) / counts.episode_agents


def _time(counts: Counts, costs: Costs) -> float:
    return (counts.steps_adjusted + costs.dist_to_goal) / counts.episodes
