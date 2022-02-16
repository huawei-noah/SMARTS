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

# to allow for typing to refer to class being defined (Mission)...
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional, Tuple

from smarts.sstudio.types import EntryTactic, TrapEntryTactic

from .coordinates import Dimensions, Heading, Point, Pose, RefLinePoint
from .road_map import RoadMap
from .utils.math import min_angles_difference_signed, vec_to_radians


class PlanningError(Exception):
    """Raised in cases when map related planning fails."""

    pass


# XXX: consider using smarts.core.coordinates.Pose for this
@dataclass(frozen=True)
class Start:
    """A starting state for a route or mission."""

    position: Tuple[int, int]
    heading: Heading
    from_front_bumper: Optional[bool] = True

    @property
    def point(self) -> Point:
        """The coordinate of this starting location."""
        return Point(*self.position)

    @classmethod
    def from_pose(cls, pose: Pose):
        """Convert to a starting location from a pose."""
        return cls(
            position=pose.position[:2],
            heading=pose.heading,
            from_front_bumper=False,
        )


@dataclass(unsafe_hash=True)
class Goal:
    """Describes an expected end state for a route or mission."""

    def is_endless(self) -> bool:
        """If the goal can never be reached."""
        return True

    def is_reached(self, vehicle) -> bool:
        """If the goal has been completed."""
        return False


@dataclass(unsafe_hash=True)
class EndlessGoal(Goal):
    """A goal that can never be completed."""

    pass


@dataclass(unsafe_hash=True)
class PositionalGoal(Goal):
    """A goal that can be completed by reaching an end area."""

    position: Point
    # target_heading: Heading
    radius: float

    @classmethod
    def from_road(
        cls,
        road_id: str,
        road_map: RoadMap,
        lane_index: int = 0,
        lane_offset: Optional[float] = None,
        radius: float = 1,
    ):
        """Generate the goal ending at the specified road lane."""
        road = road_map.road_by_id(road_id)
        lane = road.lane_at_index(lane_index)

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
        sqr_dist = (a[0] - b.x) ** 2 + (a[1] - b.y) ** 2
        return sqr_dist <= self.radius ** 2


class TraverseGoal(Goal):
    """A TraverseGoal is satisfied whenever an Agent-driven vehicle
    successfully finishes traversing a non-closed (acyclic) map
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
        pos = Point(*veh_position)
        nearest_lanes = self._road_map.nearest_lanes(pos)
        if not nearest_lanes:
            return False  # we can't tell anything here
        nl, dist = nearest_lanes[0]
        offset = nl.to_lane_coord(pos).s
        nl_width = nl.width_at_offset(offset)
        if nl.outgoing_lanes or dist < 0.5 * nl_width + 1e-1:
            return False  # the last lane it was in was not a dead-end, or it's still in a lane
        if offset < nl.length - 2 * nl_width:
            return False  # it's no where near the end of the lane
        # now check its heading to ensure it was going in roughly the right direction for this lane
        end_vec = nl.vector_at_offset(nl.length - 0.1)
        end_heading = vec_to_radians(end_vec[:2])
        heading_err = min_angles_difference_signed(end_heading, veh_heading)
        return abs(heading_err) < math.pi / 6


def default_entry_tactic(default_entry_speed: Optional[float] = None) -> EntryTactic:
    """The default tactic the simulation will use to acquire an actor for an agent."""
    return TrapEntryTactic(
        wait_to_hijack_limit_s=0,
        exclusion_prefixes=tuple(),
        zone=None,
        default_entry_speed=default_entry_speed,
    )


@dataclass(frozen=True)
class Via:
    """Describes a collectable item that can be used to shape rewards."""

    lane_id: str
    road_id: str
    lane_index: int
    position: Tuple[float, float]
    hit_distance: float
    required_speed: float


@dataclass(frozen=True)
class VehicleSpec:
    """Vehicle specifications"""

    veh_id: str
    veh_config_type: str
    dimensions: Dimensions


@dataclass(frozen=True)
class Mission:
    """A navigation mission."""

    start: Start
    goal: Goal
    # An optional list of road IDs between the start and end goal that we want to
    # ensure the mission includes
    route_vias: Tuple[str] = field(default_factory=tuple)
    start_time: float = 0.1
    entry_tactic: EntryTactic = None
    via: Tuple[Via, ...] = ()
    # if specified, will use vehicle_spec to build the vehicle (for histories)
    vehicle_spec: VehicleSpec = None

    @property
    def has_fixed_route(self) -> bool:
        """If the route is fixed and immutable."""
        return not self.goal.is_endless()

    def is_complete(self, vehicle, distance_travelled: float) -> bool:
        """If the mission has been completed successfully."""
        return self.goal.is_reached(vehicle)

    @staticmethod
    def random_endless_mission(
        road_map: RoadMap,
        min_range_along_lane: float = 0.3,
        max_range_along_lane: float = 0.9,
    ) -> Mission:
        """A mission that starts from a random location and continues indefinitely."""
        assert min_range_along_lane > 0  # Need to start further than beginning of lane
        assert max_range_along_lane < 1  # Cannot start past end of lane
        assert min_range_along_lane < max_range_along_lane  # Min must be less than max

        road = road_map.random_route(1).roads[0]
        n_lane = random.choice(road.lanes)

        # XXX: The ends of the road are not as useful as starting mission locations.
        offset = random.random() * min_range_along_lane + (
            max_range_along_lane - min_range_along_lane
        )
        offset *= n_lane.length
        coord = n_lane.from_lane_coord(RefLinePoint(offset))
        target_pose = n_lane.center_pose_at_point(coord)
        return Mission(
            start=Start(target_pose.position, target_pose.heading),
            goal=EndlessGoal(),
            entry_tactic=None,
        )


@dataclass(frozen=True)
class LapMission:
    """A mission requiring a number of laps through the goal."""

    start: Start
    goal: Goal
    route_length: float
    num_laps: int = None  # None means infinite # of laps
    # An optional list of road IDs between the start and end goal that we want to
    # ensure the mission includes
    route_vias: Tuple[str] = field(default_factory=tuple)
    start_time: float = 0.1
    entry_tactic: EntryTactic = None
    via_points: Tuple[Via, ...] = ()

    @property
    def has_fixed_route(self) -> bool:
        """If the route in this mission is immutable."""
        return True

    def is_complete(self, vehicle, distance_travelled: float) -> bool:
        """If the mission has been completed."""
        return (
            self.goal.is_reached(vehicle)
            and distance_travelled > self.route_length * self.num_laps
        )


class Plan:
    """Describes a navigation plan."""

    def __init__(
        self,
        road_map: RoadMap,
        mission: Optional[Mission] = None,
        find_route: bool = True,
    ):
        self._road_map = road_map
        self._mission = mission
        self._route = None
        if find_route:
            self.create_route(mission)

    @property
    def route(self) -> Optional[RoadMap.Route]:
        """The route that this plan calls for."""
        return self._route

    @route.setter
    def route(self, route: RoadMap.Route):
        # XXX: traffic simulator may also track route
        self._route = route

    @property
    def mission(self) -> Optional[Mission]:
        """The mission generated from this plan."""
        # XXX: This currently can be `None`
        return self._mission

    @property
    def road_map(self) -> RoadMap:
        """The road map this plan is for."""
        return self._road_map

    def create_route(self, mission: Mission) -> Mission:
        """Generates a mission that conforms to this plan."""
        assert not self._route, "already called create_route()"
        self._mission = mission or Mission.random_endless_mission(self._road_map)

        if not self._mission.has_fixed_route:
            self._route = self._road_map.empty_route()
            return self._mission

        start_lane = self._road_map.nearest_lane(
            self._mission.start.point,
            include_junctions=False,
        )
        assert start_lane, "route must start in a lane"
        start_road = start_lane.road

        end_lane = self._road_map.nearest_lane(
            self._mission.goal.position,
            include_junctions=False,
        )
        assert end_lane, "route must end in a lane"
        end_road = end_lane.road

        via_roads = [self._road_map.road_by_id(via) for via in self._mission.route_vias]

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
