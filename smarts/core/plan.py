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
# FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
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

import numpy as np

from smarts.core.coordinates import Dimensions, Heading, Point, Pose, RefLinePoint
from smarts.core.road_map import RoadMap
from smarts.core.utils.math import min_angles_difference_signed, vec_to_radians
from smarts.sstudio.types import EntryTactic, TrapEntryTactic


class PlanningError(Exception):
    """Raised in cases when map related planning fails."""

    pass


# XXX: consider using smarts.core.coordinates.Pose for this
@dataclass(frozen=True)
class Start:
    """A starting state for a route or mission."""

    position: np.ndarray
    heading: Heading
    from_front_bumper: Optional[bool] = True

    @property
    def point(self) -> Point:
        """The coordinate of this starting location."""
        return Point.from_np_array(self.position)

    @classmethod
    def from_pose(cls, pose: Pose):
        """Convert to a starting location from a pose."""
        return cls(
            position=pose.as_position2d(),
            heading=pose.heading,
            from_front_bumper=False,
        )


@dataclass(frozen=True, unsafe_hash=True)
class Goal:
    """Describes an expected end state for a route or mission."""

    def is_specific(self) -> bool:
        """If the goal is reachable at a specific position."""
        return False

    def is_reached(self, vehicle) -> bool:
        """If the goal has been completed."""
        return False


@dataclass(frozen=True, unsafe_hash=True)
class EndlessGoal(Goal):
    """A goal that can never be completed."""

    pass


@dataclass(frozen=True, unsafe_hash=True)
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

    def is_specific(self) -> bool:
        return True

    def is_reached(self, vehicle) -> bool:
        a = vehicle.position
        b = self.position
        sqr_dist = (a[0] - b.x) ** 2 + (a[1] - b.y) ** 2
        return sqr_dist <= self.radius**2


@dataclass(frozen=True, unsafe_hash=True)
class TraverseGoal(Goal):
    """A TraverseGoal is satisfied whenever an Agent-driven vehicle
    successfully finishes traversing a non-closed (acyclic) map
    It's a way for the vehicle to exit the simulation successfully,
    for example, driving across from one side to the other on a
    straight road and then continuing off the map. This goal is
    non-specific about *where* the map is exited, save for that
    the vehicle must be going the correct direction in its lane
    just prior to doing so."""

    road_map: RoadMap

    def is_specific(self) -> bool:
        return False

    def is_reached(self, vehicle) -> bool:
        pose = vehicle.pose
        return self._drove_off_map(pose.point, pose.heading)

    def _drove_off_map(self, veh_pos: Point, veh_heading: float) -> bool:
        # try to determine if the vehicle "exited" the map by driving beyond the end of a dead-end lane.
        nearest_lanes = self.road_map.nearest_lanes(veh_pos)
        if not nearest_lanes:
            return False  # we can't tell anything here
        nl, dist = nearest_lanes[0]
        offset = nl.to_lane_coord(veh_pos).s
        nl_width, conf = nl.width_at_offset(offset)
        if conf > 0.5:
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
    """A navigation mission describing a desired trip."""

    # XXX: Note that this Mission differs from sstudio.types.Mission in that
    # this can be less specific as to the particular route taken to the goal,
    # whereas sstudio.type.Mission includes a specific, predetermined/static route
    # (which might be random, but is still determined before running the scenario).

    start: Start
    goal: Goal
    # An optional list of road IDs between the start and end goal that we want to
    # ensure the mission includes
    route_vias: Tuple[str, ...] = field(default_factory=tuple)
    start_time: float = 0.1
    entry_tactic: Optional[EntryTactic] = None
    via: Tuple[Via, ...] = ()
    # if specified, will use vehicle_spec to build the vehicle (for histories)
    vehicle_spec: Optional[VehicleSpec] = None

    @property
    def requires_route(self) -> bool:
        """If the mission requires a route to be generated."""
        return self.goal.is_specific()

    def is_complete(self, vehicle, distance_travelled: float) -> bool:
        """If the mission has been completed successfully."""
        return self.goal.is_reached(vehicle)

    @staticmethod
    def endless_mission(
        start_pose: Pose,
    ) -> Mission:
        """Generate an endless mission."""
        return Mission(
            start=Start(start_pose.as_position2d(), start_pose.heading),
            goal=EndlessGoal(),
            entry_tactic=None,
        )

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
        return Mission.endless_mission(start_pose=target_pose)


@dataclass(frozen=True)
class LapMission(Mission):
    """A mission requiring a number of laps through the goal."""

    num_laps: Optional[int] = None  # None means infinite # of laps

    # If a route was specified in a sstudio.types.LapMission object,
    # then this should be set to its road length
    route_length: Optional[float] = None

    def __post_init__(self):
        # TAI: consider allowing LapMissions for TraverseGoal goals (num_laps ~ num_traversals)
        assert self.goal.is_specific
        if self.route_length is None:
            # TAI: could just assert here, but may want to be more clever...
            self.route_length = 1

    def is_complete(self, vehicle, distance_travelled: float) -> bool:
        """If the mission has been completed."""
        return (
            self.goal.is_reached(vehicle)
            and distance_travelled > self.route_length * self.num_laps
        )


class Plan:
    """Describes a navigation plan (route) to fulfill a mission."""

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
        """The mission this plan is meant to fulfill."""
        # XXX: This currently can be `None`
        return self._mission

    @property
    def road_map(self) -> RoadMap:
        """The road map this plan is relative to."""
        return self._road_map

    def create_route(self, mission: Mission, radius: Optional[float] = None):
        """Generates a route that conforms to a mission.

        Args:
            mission (Mission):
                A mission the agent should follow. Defaults to endless if `None`.
            radius (Optional[float]):
                Radius (meter) to find the nearest starting lane for the given
                mission. Defaults to `_default_lane_width` of the underlying
                road_map.
        """
        assert not self._route, "already called create_route()"
        self._mission = mission or Mission.random_endless_mission(self._road_map)

        if not self._mission.requires_route:
            self._route = self._road_map.empty_route()
            return

        assert isinstance(self._mission.goal, PositionalGoal)

        start_lanes = self._road_map.nearest_lanes(
            self._mission.start.point,
            include_junctions=True,
            radius=radius,
        )
        if not start_lanes:
            self._mission = Mission.endless_mission(Pose.origin())
            raise PlanningError("Starting lane not found. Route must start in a lane.")

        via_roads = [self._road_map.road_by_id(via) for via in self._mission.route_vias]

        end_lane = self._road_map.nearest_lane(
            self._mission.goal.position,
            include_junctions=False,
        )
        assert end_lane is not None, "route must end in a lane"

        # When an agent is in an intersection, the `nearest_lanes` method might
        # not return the correct road as the first choice. Hence, nearest
        # starting lanes are tried in sequence until a route is found or until
        # all nearby starting lane options are exhausted.
        for start_lane, _ in start_lanes:
            self._route = self._road_map.generate_routes(
                start_lane.road, end_lane.road, via_roads, 1
            )[0]
            if self._route.road_length > 0:
                break

        if len(self._route.roads) == 0:
            self._mission = Mission.endless_mission(Pose.origin())
            start_road_ids = [start_lane.road.road_id for start_lane, _ in start_lanes]
            raise PlanningError(
                "Unable to find a route between start={} and end={}. If either of "
                "these are junctions (not well supported today) please switch to "
                "roads and ensure there is a > 0 offset into the road if it is "
                "after a junction.".format(start_road_ids, end_lane.road.road_id)
            )

        return
