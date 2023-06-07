# MIT License

# Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
from __future__ import annotations

from typing import Callable, Dict, NewType

import numpy as np

from smarts.env.gymnasium.wrappers.metric.params import Params
from smarts.env.gymnasium.wrappers.metric.types import Costs, Record

Score = NewType("Score", Dict[str, float])


class FormulaBase:
    """Interface, for cost function parameters and score computation formula,
    to be implemented by other formula classes.
    """

    def __init__(self):
        pass

    def params(self) -> Params:
        """Return parameters to configure and initialize cost functions.

        Returns:
            Params: Cost function parameters.
        """
        raise NotImplementedError

    def score(self, records: Dict[str, Dict[str, Record]]) -> Score:
        """Computes sub-component scores and one total combined score named
        "Overall" on the wrapped environment.

        Args:
            records (Dict[str, Dict[str, Record]]): Records.

        Returns:
            "Overall" score and other sub-component scores.
        """
        raise NotImplementedError


class Formula(FormulaBase):
    """Sets the (i) cost function parameters, and (ii) score computation formula,
    for an environment.
    """

    def __init__(self):
        pass

    def params(self) -> Params:
        """Return parameters to configure and initialize cost functions.

        Returns:
            Params: Cost function parameters.
        """
        return Params()

    def score(self, records: Dict[str, Dict[str, Record]]) -> Score:
        """Computes sub-component scores and one total combined score named
        "Overall" on the wrapped environment.

        Args:
            records (Dict[str, Dict[str, Record]]): Records.

        Returns:
            Score: "Overall" score and other sub-component scores.
        """

        agent_weight = agent_weights(records=records)
        agent_score = agent_scores(records=records, func=costs_to_score)
        return weighted_score(scores=agent_score, weights=agent_weight)


def agent_weights(records: Dict[str, Dict[str, Record]]) -> Dict[str, Dict[str, float]]:
    """Retrieves weight for each agent in every scenario.

    Args:
        records (Dict[str, Dict[str, Record]]): Records.

    Returns:
        Dict[str,Dict[str,float]]: Weight for each agent in every scenario.
    """

    weights = {}
    for scen, agents in records.items():
        weights[scen] = dict(
            map(lambda i: (i[0], i[1].metadata.difficulty), agents.items())
        )

    return weights


def agent_scores(
    records: Dict[str, Dict[str, Record]], func: Callable[[Costs], Score]
) -> Dict[str, Dict[str, Score]]:
    """Computes score for each agent in every scenario.

    Args:
        records (Dict[str, Dict[str, Record]]): Records.
        func (Callable[[Costs],Score]): Function which computes Score given Costs.

    Returns:
        Dict[str,Dict[str,Score]]: Score for each agent in every scenario.
    """

    scores = {}
    for scen, agents in records.items():
        scores[scen] = dict(map(lambda i: (i[0], func(i[1].costs)), agents.items()))

    return scores


def weighted_score(
    scores: Dict[str, Dict[str, Score]], weights: Dict[str, Dict[str, float]]
) -> Score:
    """Computes single overall weighted score using `weights`.

    Args:
        scores (Dict[str,Dict[str,Score]]): Score for each agent in every scenario.
        weights (Dict[str,Dict[str,float]]): Weight for each agent in every scenario.

    Returns:
        Score: Weighted score.
    """
    cumulative_score = {}
    total_weight = 0
    for scen, agent in scores.items():
        for agent_name, agent_score in agent.items():
            current_score = dict(
                map(
                    lambda i: (i[0], i[1] * weights[scen][agent_name]),
                    agent_score.items(),
                )
            )
            cumulative_score = {
                score_name: score_val + cumulative_score.get(score_name, 0)
                for score_name, score_val in current_score.items()
            }
            total_weight += weights[scen][agent_name]

    return Score({key: val / total_weight for key, val in cumulative_score.items()})


def costs_to_score(costs: Costs) -> Score:
    """Compute score from costs.

    +-------------------+--------+-----------------------------------------------------------+
    |                   | Range  | Remarks                                                   |
    +===================+========+===========================================================+
    | Overall           | [0, 1] | Total score. The higher, the better.                      |
    +-------------------+--------+-----------------------------------------------------------+
    | DistToDestination | [0, 1] | Remaining distance to destination. The lower, the better. |
    +-------------------+--------+-----------------------------------------------------------+
    | Time              | [0, 1] | Time taken to complete scenario. The lower, the better.   |
    +-------------------+--------+-----------------------------------------------------------+
    | HumannessError    | [0, 1] | Humanness indicator. The lower, the better.               |
    +-------------------+--------+-----------------------------------------------------------+
    | RuleViolation     | [0, 1] | Traffic rules compliance. The lower, the better.          |
    +-------------------+--------+-----------------------------------------------------------+

    Args:
        costs (Costs): Costs.

    Returns:
        Score: Score.
    """
    dist_to_destination = costs.dist_to_destination
    humanness_error = _score_humanness_error(costs=costs)
    rule_violation = score_rule_violation(costs=costs)
    time = costs.steps
    overall = (
        0.25 * (1 - dist_to_destination)
        + 0.25 * (1 - time)
        + 0.25 * (1 - humanness_error)
        + 0.25 * (1 - rule_violation)
    )

    return Score(
        {
            "overall": overall,
            "dist_to_destination": dist_to_destination,
            "time": time,
            "humanness_error": humanness_error,
            "rule_violation": rule_violation,
        }
    )


def _score_humanness_error(costs: Costs) -> float:
    humanness_error = np.array(
        [costs.dist_to_obstacles, costs.jerk_linear, costs.lane_center_offset]
    )
    humanness_error = np.mean(humanness_error, dtype=float)
    return humanness_error


def score_rule_violation(costs: Costs) -> float:
    """Default rule violation scoring formula.

    Args:
        costs (Costs): Costs.

    Returns:
        float: Rule violation score.
    """
    rule_violation = np.array([costs.speed_limit, costs.wrong_way])
    rule_violation = np.mean(rule_violation, dtype=float)
    return rule_violation
