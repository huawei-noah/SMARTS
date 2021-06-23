# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
from dataclasses import dataclass
import random
from typing import List, Optional

import numpy as np

from .agent_interface import AgentBehavior
from .coordinates import Heading, Pose, RefLinePoint
from .road_map import RoadMap
from .scenario import EndlessGoal, Mission, Start
from .utils.math import signed_dist_to_line


class PlanningError(Exception):
    pass


@dataclass(frozen=True)
class Waypoint:
    """Dynamic, based on map and vehicle.  Unlike Sumo LanePoints,
    Waypoints do not have to snap to the middle of a road-map Lane.
    They start abreast of a vehicle's present location in the nearest Lane
    and are then interpolated along the LanePoints paths such that
    they're evenly spaced.  These are usually what is returned through
    a vehicle's sensors."""

    # XXX: consider renaming lane_id, lane_index, lane_width
    #      to nearest_lane_id, nearest_lane_index, nearest_lane_width
    pos: np.ndarray  # Point positioned on center of lane
    heading: Heading  # Heading angle of lane at this point (radians)
    lane_id: str  # ID of lane under lanepoint
    lane_width: float  # Width of lane at this point (meters)
    speed_limit: float  # Lane speed in m/s
    lane_index: int  # Index of the lane this lanepoint is over. 0 is the outer(right) most lane

    @classmethod
    def from_pose_in_lane(cls, pos: np.ndarray, heading: Heading, lane: RoadMap.Lane):
        return cls(
            pos=pos,
            heading=heading,
            lane_width=lane.width,
            speed_limit=lane.speed_limit,
            lane_id=lane.lane_id,
            lane_index=lane.index,
        )

    def __eq__(self, other):
        if not isinstance(other, Waypoint):
            return False
        return (
            (self.pos == other.pos).all()
            and self.heading == other.heading
            and self.lane_width == other.lane_width
            and self.speed_limit == other.speed_limit
            and self.lane_id == other.lane_id
            and self.lane_index == other.lane_index
        )

    def __hash__(self):
        return hash(
            (
                *self.pos,
                self.heading,
                self.lane_width,
                self.speed_limit,
                self.lane_id,
                self.lane_index,
            )
        )

    def relative_heading(self, h: Heading) -> Heading:
        """Computes relative heading between the given angle and the waypoint heading

        Returns:
            relative_heading: [-pi..pi]
        """
        assert isinstance(
            h, Heading
        ), "Heading h ({}) must be an instance of smarts.core.coordinates.Heading".format(
            type(h)
        )
        return self.heading.relative_to(h)

    def signed_lateral_error(self, p) -> float:
        """Returns the signed lateral distance from the given point to the
        line formed by the waypoint position and the waypoint heading.

        Negative signals right of line and Positive left of line.
        """
        return signed_dist_to_line(p, self.pos, self.heading.direction_vector())

    def dist_to(self, p) -> float:
        """Calculates straight line distance to the given 2D point"""
        return np.linalg.norm(self.pos - p[: len(self.pos)])


class Planner:
    def __init__(self, road_map: RoadMap, agent_behavior: AgentBehavior = None):
        self._road_map = road_map
        self._agent_behavior = agent_behavior or AgentBehavior(aggressiveness=5)
        self._mission = None
        self._route = None
        self._did_plan = False

    @property
    def route(self) -> RoadMap.Route:
        return self._route

    @property
    def mission(self) -> Mission:
        return self._mission

    @property
    def road_map(self) -> RoadMap:
        return self._road_map

    def random_endless_mission(
        self, min_range_along_lane=0.3, max_range_along_lane=0.9
    ) -> Mission:
        assert min_range_along_lane > 0  # Need to start further than beginning of lane
        assert max_range_along_lane < 1  # Cannot start past end of lane
        assert min_range_along_lane < max_range_along_lane  # Min must be less than max

        road = self._road_map.random_route(1).roads[0]
        n_lane = random.choice(road.lanes)

        # XXX: The ends of the road are not as useful as starting mission locations.
        #      Sumo complains if we get too close to 0 or `lane_length`.
        offset = random.random() * min_range_along_lane + (
            max_range_along_lane - min_range_along_lane
        )
        offset *= n_lane.length
        coord = n_lane.from_lane_coord(RefLinePoint(offset))
        target_pose = n_lane.target_pose_at_point(coord)
        return Mission(
            start=Start(target_pose.position, target_pose.heading),
            goal=EndlessGoal(),
            entry_tactic=None,
        )

    def plan(self, mission=None):  # -> Mission
        self._mission = mission or self.random_endless_mission()

        if not self._mission.has_fixed_route:
            self._route = RoadMap.Route()
        elif self._mission.task is not None:
            # TODO: ensure there is a default route
            self._route = RoadMap.Route()
        else:
            start_lane = self._road_map.nearest_lane(
                self._mission.start.point,
                include_junctions=False,
            )
            start_road = start_lane.road

            end_lane = self._road_map.nearest_lane(
                self._mission.goal.position,
                include_junctions=False,
            )
            end_road = end_lane.road

            via_roads = [
                self._road_map.road_by_id(via) for via in self._mission.route_vias
            ]

            self._route = self._road_map.generate_routes(
                start_road, end_road, via_roads, 1
            )[0]

            if len(self._route.roads) == 0:
                raise PlanningError(
                    "Unable to find a route between start={} and end={}. If either of "
                    "these are junctions (not well supported today) please switch to "
                    "roads and ensure there is a > 0 offset into the road if it's "
                    "after a junction.".format(start_road.road_id, end_road.road_id)
                )

        self._did_plan = True
        return self._mission

    def waypoint_paths(
        self,
        pose: Pose,
        lookahead: int,
        within_radius: float = 5,
        constrain_to_route: bool = True,
    ) -> List[List[Waypoint]]:
        """Computes equally-spaced Waypoints for all lane paths
        up to lookahead waypoints ahead, starting on the Road containing
        the nearest Lane aligned with the vehicle's pose within within_radius meters.
        Constrains paths to your (possibly-inferred) route only if constrain_to_route.
        Route inference assumes you're on the correct route already;
        we do not presently "replan" in case the route has changed."""
        raise NotImplementedError()

    def waypoint_paths_on_lane_at_point(
        self, pose: Pose, lane_id: str, lookahead: int, constrain_to_route: bool = True
    ) -> List[List[Waypoint]]:
        """Computes equally-spaced Waypoints for all lane paths
        up to lookahead waypoints ahead, starting at Lane lane_id
        Constrains paths to your (possibly-inferred) route only if constrain_to_route.
        Route inference assumes you're on the correct route already;
        we do not presently "replan" in case the route has changed.
        """
        raise NotImplementedError()

    def waypoint_paths_on_lane_at_offset(
        self, lane: RoadMap.Lane, offset: float, lookahead: int = 30
    ) -> List[List[Waypoint]]:
        """Computes equally-spaced Waypoints for all lane paths
        up to lookahead waypoints ahead, starting offset into lane.
        Constrains paths to your (possibly-inferred) route only if constrain_to_route.
        Route inference assumes you're on the correct route already;
        we do not presently "replan" in case the route has changed.
        """
        raise NotImplementedError()
