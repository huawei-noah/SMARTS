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

from typing import Dict

import numpy as np

from smarts.env.gymnasium.wrappers.metric.costs import Costs
from smarts.env.gymnasium.wrappers.metric.formula import (
    FormulaBase,
    Score,
    agent_scores,
    agent_weights,
    score_rule_violation,
    weighted_score,
)
from smarts.env.gymnasium.wrappers.metric.params import (
    Collisions,
    Comfort,
    DistToDestination,
    DistToObstacles,
    JerkLinear,
    LaneCenterOffset,
    OffRoad,
    Params,
    SpeedLimit,
    Steps,
    VehicleGap,
    WrongWay,
)
from smarts.env.gymnasium.wrappers.metric.types import Record


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
        params = Params(
            collisions=Collisions(active=False),
            comfort=Comfort(active=True),
            dist_to_destination=DistToDestination(active=True),
            dist_to_obstacles=DistToObstacles(active=False),
            jerk_linear=JerkLinear(active=False),
            lane_center_offset=LaneCenterOffset(active=True),
            off_road=OffRoad(active=False),
            speed_limit=SpeedLimit(active=True),
            steps=Steps(active=False),
            vehicle_gap=VehicleGap(active=True),
            wrong_way=WrongWay(active=True),
        )
        return params

    def score(self, records: Dict[str, Dict[str, Record]]) -> Score:
        """
        Computes several sub-component scores and one total combined score named
        "Overall" on the wrapped environment.

        Args:
            records (Dict[str, Dict[str, Record]]): Records.

        Returns:
            Score: "Overall" score and other sub-component scores.
        """

        agent_weight = agent_weights(records=records)
        agent_score = agent_scores(records=records, func=costs_to_score)
        return weighted_score(scores=agent_score, weights=agent_weight)


def costs_to_score(costs: Costs) -> Score:
    """Compute score from costs.

    +-------------------+--------+-----------------------------------------------------------+
    |                   | Range  | Remarks                                                   |
    +===================+========+===========================================================+
    | Overall           | [0, 1] | Total score. The higher, the better.                      |
    +-------------------+--------+-----------------------------------------------------------+
    | DistToDestination | [0, 1] | Remaining distance to destination. The lower, the better. |
    +-------------------+--------+-----------------------------------------------------------+
    | VehicleGap        | [0, 1] | Gap between vehicles in a convoy. The lower, the better.  |
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
    vehicle_gap = costs.vehicle_gap
    overall = (
        0.25 * (1 - dist_to_destination)
        + 0.25 * (1 - vehicle_gap)
        + 0.25 * (1 - humanness_error)
        + 0.25 * (1 - rule_violation)
    )

    return Score(
        {
            "overall": overall,
            "dist_to_destination": dist_to_destination,
            "vehicle_gap": vehicle_gap,
            "humanness_error": humanness_error,
            "rule_violation": rule_violation,
        }
    )


def _score_humanness_error(costs: Costs) -> float:
    humanness_error = np.array([costs.comfort, costs.lane_center_offset])
    humanness_error = np.mean(humanness_error, dtype=float)
    return humanness_error
