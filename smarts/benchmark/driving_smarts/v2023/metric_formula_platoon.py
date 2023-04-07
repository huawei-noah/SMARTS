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
from dataclasses import fields
from typing import Dict, TypeVar

import numpy as np

from smarts.env.gymnasium.wrappers.metric.costs import Costs
from smarts.env.gymnasium.wrappers.metric.counts import Counts
from smarts.env.gymnasium.wrappers.metric.formula import FormulaBase, Score
from smarts.env.gymnasium.wrappers.metric.params import (
    Comfort,
    DistToDestination,
    DistToObstacles,
    GapBetweenVehicles,
    Params,
)
from smarts.env.gymnasium.wrappers.metric.types import Record
from smarts.env.gymnasium.wrappers.metric.utils import add_dataclass, op_dataclass, multiply, divide


class Formula(FormulaBase):
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
            ),
            dist_to_destination=DistToDestination(
                active=True,
                wrt="Leader-007",
            ),
            dist_to_obstacles=DistToObstacles(
                active=True,
                ignore=[
                    "ego",
                    "Leader-007",
                ],  # <------- Ego is not ignored yet !!!!!!!!!!!
            ),
            gap_between_vehicles=GapBetweenVehicles(
                active=False,
                interest="Leader-007",
            ),
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
            Dict[str, float]: Contains "Overall", "DistToDestination", "Time",
            "Humanness", and "Rules" scores.
        """

        counts_list, costs_list = zip(
            *[
                (record.counts, record.costs)
                for scen, val in records_sum.items()
                for agent, record in val.items()
            ]
        ) # <--- check correctness of averaging here

        agents_tot: int = len(counts_list)  # Total number of agents over all scenarios
        counts_tot: Counts = functools.reduce(
            lambda a, b: add_dataclass(a, b), counts_list
        )
        costs_tot: Costs = functools.reduce(
            lambda a, b: add_dataclass(a, b), costs_list
        )

        # <------ Check summing of different values according to intended formula
        dist_to_destination = _dist_to_destination(costs=costs_tot, agents_tot=agents_tot)
        humanness = _humanness(costs=costs_tot, agents_tot=agents_tot)
        rules = _rules(costs=costs_tot, agents_tot=agents_tot)
        time = _time(counts=counts_tot)
        overall = (
            0.35 * (1 - dist_to_destination)
            + 0.30 * (1 - time)
            + 0.25 * humanness
            + 0.10 * rules
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


def _dist_to_destination(costs:Costs, agents_tot:int) -> float:
    return costs.dist_to_destination / agents_tot


def _humanness(costs: Costs, agents_tot: int) -> float:
    humanness_to_minimize = np.array(
        [costs.dist_to_obstacles, costs.jerk_linear, costs.lane_center_offset]
    )
    humanness_to_minimize = np.mean(humanness_to_minimize, dtype=float) / agents_tot
    return 1 - humanness_to_minimize


def _rules(costs: Costs, agents_tot: int) -> float:
    rules_to_minimize = np.array([costs.speed_limit, costs.wrong_way])
    rules_to_minimize = np.mean(rules_to_minimize, dtype=float) / agents_tot
    return 1 - rules_to_minimize


def _time(counts: Counts) -> float:
    return counts.steps_adjusted / counts.max_steps
