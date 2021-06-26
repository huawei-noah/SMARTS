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
from dataclasses import dataclass, field
import math
import random
from typing import List, Optional, Tuple

import numpy as np

from .coordinates import Dimensions, Heading, Point, Pose, RefLinePoint
from .road_map import RoadMap
from .utils.math import signed_dist_to_line

from smarts.sstudio.types import EntryTactic, TrapEntryTactic


class PlanningError(Exception):
    pass


# XXX: consider using smarts.core.coordinates.Pose for this
@dataclass(frozen=True)
class Start:
    position: Tuple[int, int]
    heading: Heading

    @property
    def point(self) -> Point:
        return Point(*self.position)


@dataclass(frozen=True)
class Goal:
    def is_endless(self) -> bool:
        return True

    def is_reached(self, vehicle) -> bool:
        return False


@dataclass(frozen=True)
class EndlessGoal(Goal):
    pass


@dataclass(frozen=True)
class PositionalGoal(Goal):
    position: Point
    # target_heading: Heading
    radius: float

    @classmethod
    def fromedge(cls, road_id, road_map, lane_index=0, lane_offset=None, radius=1):
        road = road_map.road_by_id(road_id)
        lane = road.lane_at_index(lane_index)  # XXX: bidirectional roads?

        if lane_offset is None:
            # Default to the midpoint safely ensuring we are on the lane and not
            # bordering another
            lane_offset = lane.length * 0.5

        position = lane.from_lane_coord(RefLinePoint(lane_offset))
        return cls(position=position, radius=radius)

    def is_endless(self) -> bool:
        return False

    def is_reached(self, vehicle) -> bool:
        a = vehicle.position
        b = self.position
        dist = math.sqrt((a[0] - b.x) ** 2 + (a[1] - b.y) ** 2)
        return dist <= self.radius


@dataclass
class TraverseGoal(Goal):
    """A TraverseGoal is satisfied whenever an Agent-driven vehicle
    successfully finishes traversing a non-closed (acyclical) map
    It's a way for the vehicle to exit the simulation successfully,
    for example, driving across from one side to the other on a
    straight road and then continuing off the map.  This goal is
    non-specific about *where* the map is exited, save for that
    the vehicle must be going the correct direction in its lane
    just prior to doing so."""

    def __init__(self, road_map: RoadMap):
        super().__init__()
        self._road_map = road_map

    def is_endless(self) -> bool:
        return True

    def is_reached(self, vehicle) -> bool:
        return self._drove_off_map(vehicle.position, vehicle.heading)

    def _drove_off_map(
        self, veh_position: Tuple[float, float, float], veh_heading: float
    ) -> bool:
        # try to determine if the vehicle "exited" the map by driving beyond the end of a dead-end lane.
        nearest_lanes = self._road_map.nearest_lanes(veh_position)
        if not nearest_lanes:
            return False  # we can't tell anything here
        nl, dist = nearest_lanes[0]
        # TODO STEVE:  nl.width_at_offset!
        if nl.outgoing_lanes or dist < 0.5 * nl.width + 1e-1:
            return False  # the last lane it was in was not a dead-end, or it's still in a lane
        end_node = nl.road.getToNode()  # TODO SUMO road_network
        end_point = end_node.getCoord()
        dist = math.sqrt(
            (veh_position[0] - end_point[0]) ** 2
            + (veh_position[1] - end_point[1]) ** 2
        )
        if dist > 2 * nl.width:
            return False  # it's no where near the end of the lane
        # now check its heading to ensure it was going in roughly the right direction for this lane
        end_shape = end_node.getShape()
        veh_heading %= 2 * math.pi
        tolerance = math.pi / 4
        for p in range(1, len(end_shape)):
            num = end_shape[p][1] - end_shape[p - 1][1]
            den = end_shape[p][0] - end_shape[p - 1][0]
            crossing_heading = math.atan(-den / num)
            if den < 0:
                crossing_heading += math.pi
            elif num < 0:
                crossing_heading -= math.pi
            crossing_heading -= math.pi / 2
            # we allow for it to be going either way since it's a pain to determine which side of the edge it's on
            if (
                abs(veh_heading - crossing_heading % (2 * math.pi)) < tolerance
                or abs(
                    (veh_heading + math.pi) % (2 * math.pi)
                    - crossing_heading % (2 * math.pi)
                )
                < tolerance
            ):
                return True
        return False


def default_entry_tactic(default_entry_speed: float = None) -> EntryTactic:
    return TrapEntryTactic(
        wait_to_hijack_limit_s=0,
        exclusion_prefixes=tuple(),
        zone=None,
        default_entry_speed=default_entry_speed,
    )


@dataclass(frozen=True)
class Via:
    lane_id: str
    edge_id: str
    lane_index: int
    position: Tuple[float, float]
    hit_distance: float
    required_speed: float


@dataclass(frozen=True)
class VehicleSpec:
    veh_id: str
    veh_type: str
    dimensions: Dimensions


@dataclass(frozen=True)
class Mission:
    start: Start
    goal: Goal
    # An optional list of edge IDs between the start and end goal that we want to
    # ensure the mission includes
    route_vias: Tuple[str] = field(default_factory=tuple)
    start_time: float = 0.1
    entry_tactic: EntryTactic = None
    via: Tuple[Via, ...] = ()
    # if specified, will use vehicle_spec to build the vehicle (for histories)
    vehicle_spec: VehicleSpec = None

    @property
    def has_fixed_route(self) -> bool:
        return not self.goal.is_endless()

    def is_complete(self, vehicle, distance_travelled: float) -> bool:
        return self.goal.is_reached(vehicle)


@dataclass(frozen=True)
class LapMission:
    start: Start
    goal: Goal
    route_length: float
    num_laps: int = None  # None means infinite # of laps
    # An optional list of edge IDs between the start and end goal that we want to
    # ensure the mission includes
    route_vias: Tuple[str] = field(default_factory=tuple)
    start_time: float = 0.1
    entry_tactic: EntryTactic = None
    via_points: Tuple[Via, ...] = ()

    @property
    def has_fixed_route(self) -> bool:
        return True

    def is_complete(self, vehicle, distance_travelled: float) -> bool:
        return (
            self.goal.is_reached(vehicle)
            and distance_travelled > self.route_length * self.num_laps
        )


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
    def __init__(self, road_map: RoadMap):
        self._road_map = road_map
        self._mission = None
        self._route = None

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
