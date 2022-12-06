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

from dataclasses import dataclass
from typing import Callable
from smarts.core.plan import Mission, Plan, PositionalGoal
from smarts.core.road_map import RoadMap
from smarts.core.utils.math import running_mean
from smarts.core.sensors import Observation

@dataclass(frozen=True)
class Completion:
    """Parameters to compute the percentage of scenario tasks completed."""

    dist_remainder: float = 0
    """The remaining distance to reach goal position. Computed using the
    shortest route distance between current position and goal position.
    """
    dist_tot: float = 0
    """ Shortest route distance between start and goal position.
    """

def get_dist(road_map:RoadMap, mission: Mission) -> float:
    """
    Returns the shortest route distance between the start and end point given
    in the mission.

    Args:
        road_map (RoadMap): Scenario road map
        mission (Mission): A mission, with PositionalGoal, stating its start
            and end positions.

    Returns:
        float: Shortest route distance from start to end position.
    """
    plan = Plan(road_map, mission, find_route=True)
    route = plan.route
    route_point_start = RoadMap.Route.RoutePoint(pt=mission.start.point)
    route_point_end = RoadMap.Route.RoutePoint(pt=mission.goal.position)
    dist_tot = route.distance_between(start=route_point_start, end=route_point_end)
    return dist_tot

    # print(f"{agent_name} Starting point -- {agent_mission.start.point}, {type(agent_mission.start.point)}")
    # print(f"{agent_name} Goal point -- {agent_mission.goal.position}, {type(agent_mission.goal.position)}")
    # for roads in agent_plan[agent_name].route.roads:
    #     print(f"{roads.road_id}")
    # print(f"Route Length: {agent_plan[agent_name].route.road_length}")

def _dist_remainder():
    mean:float = 0
    step:int = 0

    def func(road_map:RoadMap, completion:Completion, obs: Observation):
        nonlocal mean, step

        mission = Mission(
            start=obs.ego_vehicle_state.position,
            goal=PositionalGoal(
                
            )
        )

        dist = get_dist(road_map=road_map, mission=mission)
        mean, step = running_mean(prev_mean=mean, prev_step=step, new_val=dist)
        return Completion(dist_remainder=mean)

    return func

def _dist_tot(road_map:RoadMap, completion:Completion,obs:Observation)->Completion:
    return Completion(dist_tot=completion.dist_tot)


class CompletionFuncs:
    """Functions to compute scenario completion metrics."""

    dist_tot: Callable[[RoadMap, Completion, Observation], Completion] = _dist_tot
    distance: Callable[[RoadMap, Completion, Observation], Completion] = _dist_remainder  
