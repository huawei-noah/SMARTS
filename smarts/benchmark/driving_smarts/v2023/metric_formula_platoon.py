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
    DistToDestination,
    DistToObstacles,
    GapBetweenVehicles,
    Params,
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
                active=False,
            ),  # <------------------ Not implemented yet !!!!!!!!!!!
            dist_to_obstacles=DistToObstacles(
                active=True,
                ignore=[
                    "ego",
                    "Leader-007",
                ],  # <---------------- Ego is not ignored yet !!!!!!!!!!!
            ),
            gap_between_vehicles=GapBetweenVehicles(
                active=False,
                interest="Leader-007",
            ),  # <------------------ Not implemented yet !!!!!!!!!!!
        )
        return params

    def score(self, records_sum: Dict[str, Dict[str, Record]]) -> Score:
        """
        Computes four sub-component scores, namely, "Distance to Destination",
        "Time", "Humanness", "Rules", and one total combined score named
        "Overall" on the wrapped environment.

        +-------------------+--------+-----------------------------------------------------------+
        |                   | Range  | Remarks                                                   |
        +===================+========+===========================================================+
        | Overall           | [0, 1] | Total score. The higher, the better.                      |
        +-------------------+--------+-----------------------------------------------------------+
        | DistToDestination | [0, 1] | Remaining distance to destination. The lower, the better. |
        +-------------------+--------+-----------------------------------------------------------+
        | Time              | [0, 1] | Time taken to complete scenario. The lower, the better.   |
        +-------------------+--------+-----------------------------------------------------------+
        | Humanness         | [0, 1] | Humanness indicator. The higher, the better.              |
        +-------------------+--------+-----------------------------------------------------------+
        | Rules             | [0, 1] | Traffic rules compliance. The higher, the better.         |
        +-------------------+--------+-----------------------------------------------------------+

        Returns:
            Score: Contains "Overall", "DistToDestination", "Time",
            "Humanness", and "Rules" scores.
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
        humanness = _humanness(costs=costs_final)
        rules = _rules(costs=costs_final)
        time = costs_final.steps
        overall = (
            0.50 * (1 - dist_to_destination)
            + 0.25 * (1 - time)
            + 0.20 * humanness
            + 0.05 * rules
        )

        return Score(
            {
                "overall": overall,
                "dist_to_destination": dist_to_destination,
                "time": time,
                "humanness": humanness,
                "rules": rules,
            }
        )


def _humanness(costs: Costs) -> float:
    humanness = np.array(
        [costs.dist_to_obstacles, costs.jerk_linear, costs.lane_center_offset]
    )
    humanness = np.mean(humanness, dtype=float)
    return 1 - humanness


def _rules(costs: Costs) -> float:
    rules = np.array([costs.speed_limit, costs.wrong_way])
    rules = np.mean(rules, dtype=float)
    return 1 - rules
