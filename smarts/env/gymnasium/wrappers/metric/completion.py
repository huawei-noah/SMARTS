# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import logging
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict

from smarts.core.coordinates import Heading, Point
from smarts.core.observations import Observation
from smarts.core.plan import Mission, Plan, PlanningError, PositionalGoal, Start
from smarts.core.road_map import RoadMap
from smarts.core.utils.math import running_mean

logger = logging.getLogger(__file__)

MAXIMUM_OFFROAD_DIST = 300


@dataclass(frozen=True)
class Completion:
    """Parameters to compute the percentage of scenario tasks completed."""

    dist_remainder: float = 0
    """Shortest road distance between current position and goal position.
    """
    dist_tot: float = 0
    """Shortest route distance between start position and goal position.
    """


class CompletionError(Exception):
    """Raised when computation of `Completion` metric fails."""

    pass


def get_dist(road_map: RoadMap, point_a: Point, point_b: Point) -> float:
    """
    Computes the shortest route distance from point_a to point_b in the road
    map. If no routes are available from point_a to point_b, then distance of
    the shortest road length from point_b to point_a is instead computed. Both
    points should lie on a road in the road map.

    Args:
        road_map: Scenario road map.
        point_a: A point, in world-map coordinates, which lies on a road.
        point_b: A point, in world-map coordinates, which lies on a road.

    Returns:
        float: Shortest road distance between two points in the road map.
    """

    def _get_dist(start: Point, end: Point) -> float:
        mission = Mission(
            start=Start(
                position=start.as_np_array,
                heading=Heading(0),
                from_front_bumper=False,
            ),
            goal=PositionalGoal(
                position=end,
                radius=3,
            ),
        )
        plan = Plan(road_map=road_map, mission=mission, find_route=False)
        plan.create_route(mission=mission, radius=MAXIMUM_OFFROAD_DIST)
        from_route_point = RoadMap.Route.RoutePoint(pt=start)
        to_route_point = RoadMap.Route.RoutePoint(pt=end)

        dist_tot = plan.route.distance_between(
            start=from_route_point, end=to_route_point
        )
        if dist_tot == None:
            raise CompletionError("Unable to find road on route near given points.")
        elif dist_tot < 0:
            raise CompletionError(
                "Path from start point to end point flows in "
                "the opposite direction of the generated route."
            )

        return dist_tot

    try:
        dist_tot = _get_dist(point_a, point_b)
    except PlanningError as err:
        if err.args[0].startswith("Unable to find a route"):
            # Vehicle might end (i) in a one-way road, or (ii) in a road without
            # u-turn, causing the route planner to fail. When there is no legal
            # route, the walkable road distance in the reverse direction is
            # returned as the distance between point_a and point_b.
            dist_tot = _get_dist(point_b, point_a)
            logger.info(
                "completion.get dist(): Did not find a route from "
                "%s to %s, instead found a reversed route from %s to %s.",
                point_a,
                point_b,
                point_b,
                point_a,
            )
        else:
            raise
    except CompletionError:
        dist_tot = 1e10
        warnings.warn(
            "completion.get dist(): Did not find a route from "
            f"{point_a} to {point_b}, because too far off road. "
            f"Agent vehicle is more than {MAXIMUM_OFFROAD_DIST}m off road."
            "This will cause a large penalty in completion score.",
        )
    return dist_tot


def _dist_remainder():
    mean: float = 0
    step: int = 0

    def func(
        road_map: RoadMap, obs: Observation, initial_compl: Completion
    ) -> Completion:
        nonlocal mean, step

        if obs.events.reached_goal:
            dist = 0
        else:
            cur_pos = Point(*obs.ego_vehicle_state.position)
            goal_position = getattr(obs.ego_vehicle_state.mission.goal, "position")
            assert (
                goal_position is not None
            ), f"Ego `{obs.ego_vehicle_state.id}` cannot use cost func without a positional goal."
            goal_point = Point(*goal_position)
            dist = get_dist(road_map=road_map, point_a=cur_pos, point_b=goal_point)

        # Cap remainder distance
        c_dist = min(dist, initial_compl.dist_tot)

        mean, step = running_mean(prev_mean=mean, prev_step=step, new_val=c_dist)
        return Completion(dist_remainder=mean)

    return func


@dataclass(frozen=True)
class CompletionFuncs:
    """Functions to compute scenario completion metrics. Each function computes
    the running mean completion value over number of episodes, for a given
    scenario."""

    # fmt: off
    dist_remainder: Callable[[RoadMap, Observation, Completion], Completion] = _dist_remainder()
    # fmt: on
