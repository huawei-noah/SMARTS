import functools
from dataclasses import fields
from typing import Dict, NewType, TypeVar

import numpy as np

from smarts.env.gymnasium.wrappers.metric.completion import Completion
from smarts.env.gymnasium.wrappers.metric.costs import Costs
from smarts.env.gymnasium.wrappers.metric.counts import Counts
from smarts.env.gymnasium.wrappers.metric.params import Params
from smarts.env.gymnasium.wrappers.metric.types import Data

Score = NewType("Score", Dict[str, float])


class FormulaBase:
    def __init__(self):
        pass

    def params(self) -> Params:
        raise NotImplementedError

    def score(self) -> Score:
        raise NotImplementedError


class Formula(FormulaBase):
    def __init__(self):
        pass

    def params(self) -> Params:
        return Params()

    def score(self, records: Dict[str, Data]) -> Score:
        """
        Computes four sub-component scores, namely, "Completion", "Time",
        "Humanness", "Rules", and one total combined score named "Overall"
        on the wrapped environment.

        Describes the final score given by processing observations through the metrics.

            +-------------+--------+-----------------------------------------------------------------------------------------------------+
            |             | Range  | Remarks                                                                                             |
            +=============+========+=====================================================================================================+
            | Overall     | [0, 1] | Total score which combines "Completion", "Time", "Humanness", and "Rules". The higher, the better.  |
            +-------------+--------+-----------------------------------------------------------------------------------------------------+
            | Completion  | [0, 1] | Proportion of scenarios tasks completed. The higher, the better.                                    |
            +-------------+--------+-----------------------------------------------------------------------------------------------------+
            | Time        | [0, 1] | Time taken to complete scenario. The lower, the better.                                             |
            +-------------+--------+-----------------------------------------------------------------------------------------------------+
            | Humanness   | [0, 1] | Humanness indicator. The higher, the better.                                                        |
            +-------------+--------+-----------------------------------------------------------------------------------------------------+
            | Rules       | [0, 1] | Traffic rules compliance. The higher, the better.                                                   |
            +-------------+--------+-----------------------------------------------------------------------------------------------------+

        Returns:
            Dict[str, float]: Contains "Overall", "Completion", "Time",
            "Humanness", and "Rules" scores.
        """

        counts_list, costs_list, completion_list = zip(
            *[
                (data.record.counts, data.record.costs, data.record.completion)
                for agents in records.values()
                for data in agents.values()
            ]
        )
        agents_tot: int = len(counts_list)  # Total number of agents over all scenarios
        counts_tot: Counts = functools.reduce(
            lambda a, b: _add_dataclass(a, b), counts_list
        )
        costs_tot: Costs = functools.reduce(
            lambda a, b: _add_dataclass(a, b), costs_list
        )
        completion_tot: Completion = functools.reduce(
            lambda a, b: _add_dataclass(a, b), completion_list
        )

        completion = _completion(completion=completion_tot)
        humanness = _humanness(costs=costs_tot, agents_tot=agents_tot)
        rules = _rules(costs=costs_tot, agents_tot=agents_tot)
        time = _time(counts=counts_tot)
        overall = completion * (1 - time) * humanness * rules

        return Score(
            {
                "dist_to_completion": completion,
                "humanness": humanness,
                "rules": rules,
                "time": time,
                "overall": overall,
            }
        )


T = TypeVar("T", Completion, Costs, Counts)


def _add_dataclass(first: T, second: T) -> T:
    assert type(first) is type(second)
    new = {}
    for field in fields(first):
        sum = getattr(first, field.name) + getattr(second, field.name)
        new[field.name] = sum
    output = first.__class__(**new)

    return output


def _completion(completion: Completion) -> float:
    """
    Proportion of scenarios tasks completed.

    Args:
        completion (Completion): Scenario tasks completed.

    Returns:
        float: Normalized completion value = [0, 1]. Completion value should be
            maximized. The higher the value, the better it is.
    """
    return (completion.dist_tot - completion.dist_remainder) / completion.dist_tot


def _humanness(costs: Costs, agents_tot: int) -> float:
    """
    Humanness indicator.

    Args:
        costs (Costs): Performance cost values.
        agents_tot (int): Number of agents simulated.

    Returns:
        float: Normalized humanness value = [0, 1]. Humanness value should be
            maximized. The higher the value, the better it is.
    """
    humanness_to_minimize = np.array(
        [costs.dist_to_obstacles, costs.jerk_linear, costs.lane_center_offset]
    )
    humanness_to_minimize = np.mean(humanness_to_minimize, dtype=float) / agents_tot
    return 1 - humanness_to_minimize


def _rules(costs: Costs, agents_tot: int) -> float:
    """
    Traffic rules compliance.

    Args:
        costs (Costs): Performance cost values.
        agents_tot (int): Number of agents simulated.

    Returns:
        float: Normalized rules value = [0, 1]. Rules value should be maximized.
            The higher the value, the better it is.
    """
    rules_to_minimize = np.array([costs.speed_limit, costs.wrong_way])
    rules_to_minimize = np.mean(rules_to_minimize, dtype=float) / agents_tot
    return 1 - rules_to_minimize


def _time(counts: Counts) -> float:
    """
    Time taken to complete scenario.

    Args:
        counts (Counts): Performance count values.

    Returns:
        float: Normalized time value = (0, 1]. Time value should be minimized.
            The lower the value, the better it is.
    """
    return counts.steps_adjusted / counts.max_steps
