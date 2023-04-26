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

import functools
from typing import Dict

import numpy as np

from smarts.env.gymnasium.wrappers.metric.costs import Costs
from smarts.env.gymnasium.wrappers.metric.formula import FormulaBase, Score
from smarts.env.gymnasium.wrappers.metric.params import (
    Comfort,
    DistToObstacles,
    JerkLinear,
    Params,
    Steps,
    VehicleGap,
)
from smarts.env.gymnasium.wrappers.metric.types import Record
from smarts.env.gymnasium.wrappers.metric.utils import (
    add_dataclass,
    divide,
    op_dataclass,
)


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
            comfort=Comfort(
                active=True,
            ),
            dist_to_obstacles=DistToObstacles(
                active=False,
            ),
            jerk_linear=JerkLinear(active=False),
            vehicle_gap=VehicleGap(
                active=True,
                actor="Leader-007",
            ),
            steps=Steps(
                active=False,
            ),
        )
        return params

    def score(self, records_sum: Dict[str, Dict[str, Record]]) -> Score:
        """
        Computes several sub-component scores and one total combined score named
        "Overall" on the wrapped environment.

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

        Returns:
            Score: Contains "Overall", "DistToDestination", "VehicleGap",
            "HumannessError", and "RuleViolation" scores.
        """

        costs_total = Costs()
        episodes = 0
        for scen, val in records_sum.items():
            # Number of agents in scenario.
            agents_in_scenario = len(val.keys())
            costs_list, counts_list = zip(
                *[(record.costs, record.counts) for agent, record in val.items()]
            )
            # Sum costs over all agents in scenario.
            costs_sum_agent: Costs = functools.reduce(
                lambda a, b: add_dataclass(a, b), costs_list
            )
            # Average costs over number of agents in scenario.
            costs_mean_agent = op_dataclass(costs_sum_agent, agents_in_scenario, divide)
            # Sum costs over all scenarios.
            costs_total = add_dataclass(costs_total, costs_mean_agent)
            # Increment total number of episodes.
            episodes += counts_list[0].episodes

        # Average costs over total number of episodes.
        costs_final = op_dataclass(costs_total, episodes, divide)

        # Compute sub-components of score.
        dist_to_destination = costs_final.dist_to_destination
        humanness_error = _humanness_error(costs=costs_final)
        rule_violation = _rule_violation(costs=costs_final)
        vehicle_gap = costs_final.vehicle_gap
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


def _humanness_error(costs: Costs) -> float:
    humanness_error = np.array([costs.comfort, costs.lane_center_offset])
    humanness_error = np.mean(humanness_error, dtype=float)
    return humanness_error


def _rule_violation(costs: Costs) -> float:
    rule_violation = np.array([costs.speed_limit, costs.wrong_way])
    rule_violation = np.mean(rule_violation, dtype=float)
    return rule_violation
