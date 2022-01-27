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

# to allow for typing to refer to class being defined (RoadMap)
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Set, Tuple

import numpy as np
from shapely.geometry import Polygon

from .coordinates import BoundingBox, Heading, Point, Pose, RefLinePoint
from .utils.math import (
    fast_quaternion_from_angle,
    min_angles_difference_signed,
    signed_dist_to_line,
    vec_to_radians,
)

# TODO:
# - also consider Esri, QGIS and Google Maps formats
# -https://www.asam.net/index.php?eID=dumpFile&t=f&f=4089&token=deea5d707e2d0edeeb4fccd544a973de4bc46a09


class RoadMap:
    """Base class from which map implementation classes extend."""

    @property
    def source(self) -> str:
        """The road map resource source. Generally a file URI."""
        raise NotImplementedError()

    @property
    def bounding_box(self) -> Optional[BoundingBox]:
        """The minimum bounding box that contains the map geometry. May return `None` to indicate
        the map is unbounded.
        """
        raise NotImplementedError()

    @property
    def scale_factor(self) -> float:
        """The ratio between 1 unit on the map and 1 meter."""
        # map units per meter
        return 1.0

    def is_same_map(self, map_spec) -> bool:
        """Check if the MapSpec Object source points to the same RoadMap instance as the current"""
        raise NotImplementedError

    def to_glb(self, at_path: str):
        """Build a glb file for camera rendering and envision"""
        raise NotImplementedError()

    def surface_by_id(self, surface_id: str) -> RoadMap.Surface:
        """Find a surface within the road map that has the given identifier."""
        raise NotImplementedError()

    def lane_by_id(self, lane_id: str) -> RoadMap.Lane:
        """Find a lane in this road map that has the given identifier."""
        raise NotImplementedError()

    def road_by_id(self, road_id: str) -> RoadMap.Road:
        """Find a road in this road map that has the given identifier."""
        raise NotImplementedError()

    def nearest_lanes(
        self, point: Point, radius: Optional[float] = None, include_junctions=True
    ) -> List[Tuple[RoadMap.Lane, float]]:
        """Find lanes on this road map that are near the given point."""
        raise NotImplementedError()

    def nearest_lane(
        self, point: Point, radius: Optional[float] = None, include_junctions=True
    ) -> RoadMap.Lane:
        """Find the nearest lane on this road map to the given point."""
        nearest_lanes = self.nearest_lanes(point, radius, include_junctions)
        return nearest_lanes[0][0] if nearest_lanes else None

    def road_with_point(self, point: Point) -> RoadMap.Road:
        """Find the road that contains the given point."""
        raise NotImplementedError()

    def generate_routes(
        self,
        start_road: RoadMap.Road,
        end_road: RoadMap.Road,
        via: Optional[Sequence[RoadMap.Road]] = None,
        max_to_gen: int = 1,
    ) -> List[RoadMap.Route]:
        """Generates routes between two roads.
        Args:
            start_road:
                The beginning road of the generated routes.
            end_road:
                The end road of the generated routes.
            via:
                All edges that the generated routes must pass through.
            max_to_gen:
                The maximum number of routes to generate.
        Returns:
            A list of generated routes that satisfy the given restrictions. Routes will be
             returned in order of increasing length.
        """
        raise NotImplementedError()

    def random_route(self, max_route_len: int = 10) -> RoadMap.Route:
        """Generate a random route contained in this road map.
        Args:
            max_route_len:
                The total number of roads in the route.
        Returns:
            A randomly generated route.
        """
        raise NotImplementedError()

    def empty_route(self) -> RoadMap.Route:
        """Generate an empty route."""
        raise NotImplementedError()

    def waypoint_paths(
        self,
        pose: Pose,
        lookahead: int,
        within_radius: float = 5,
        route: RoadMap.Route = None,
    ) -> List[List[Waypoint]]:
        """Computes equally-spaced Waypoints for all lane paths
        up to lookahead waypoints ahead, starting on the Road containing
        the nearest Lane aligned with the vehicle's pose within within_radius meters.
        Constrains paths to the supplied route if specified."""
        raise NotImplementedError()

    class Surface:
        """Describes a surface."""

        @property
        def surface_id(self) -> str:
            """The unique identifier for a surface."""
            raise NotImplementedError()

        @property
        def is_drivable(self) -> bool:
            """Returns true iff this surface is legally and physically drivable."""
            raise NotImplementedError()

        @property
        def entry_surfaces(self) -> List[RoadMap.Surface]:
            """Surfaces by which one might enter this surface."""
            raise NotImplementedError()

        @property
        def exit_surfaces(self) -> List[RoadMap.Surface]:
            """Surfaces by which one might exit this surface."""
            raise NotImplementedError()

        @property
        def features(self) -> List[RoadMap.Feature]:
            """The features that this surface contains."""
            raise NotImplementedError()

        def features_near(self, pose: Pose, radius: float) -> List[RoadMap.Feature]:
            """The features on this surface near the given pose."""
            raise NotImplementedError()

        def shape(
            self, buffer_width: float = 0.0, default_width: Optional[float] = None
        ) -> Polygon:
            """Returns a convex polygon representing this surface, buffered by buffered_width (which must be non-negative),
            where buffer_width is a buffer around the perimeter of the polygon.  In some situations, it may be desirable to
            also specify a `default_width`, in which case the returned polygon should have a convex shape where the
            distance across it is no less than buffered_width + default_width at any point."""
            raise NotImplementedError()

        def contains_point(self, point: Point) -> bool:
            """Returns True iff this point is fully contained by this surface."""
            raise NotImplementedError()

    class Lane(Surface):
        """Describes a lane surface."""

        @property
        def lane_id(self) -> str:
            """Unique identifier for this Lane."""
            raise NotImplementedError()

        @property
        def road(self) -> RoadMap.Road:
            """The road that this lane is a part of."""
            raise NotImplementedError()

        @property
        def composite_lane(self) -> RoadMap.Lane:
            """Return an abstract Lane composed of one or more RoadMap.Lane segments
            that has been inferred to correspond to one continuous real-world lane.
            May return same object as self."""
            return self

        @property
        def is_composite(self) -> bool:
            """Return True if this Lane object was inferred
            and composed out of subordinate Lane objects."""
            return False

        @property
        def speed_limit(self) -> float:
            """The speed limit on this lane."""
            raise NotImplementedError()

        @property
        def length(self) -> float:
            """The length of this lane."""
            raise NotImplementedError()

        @property
        def in_junction(self) -> bool:
            """If this lane is a part of a junction (usually an intersection.)"""
            raise NotImplementedError()

        @property
        def index(self) -> int:
            """when not in_junction, 0 is outer / right-most (relative to lane heading) lane on road.
            otherwise, index scheme is implementation-dependent, but must be deterministic."""
            # TAI:  UK roads
            raise NotImplementedError()

        @property
        def lanes_in_same_direction(self) -> List[RoadMap.Lane]:
            """returns all other lanes on this road where traffic goes
            in the same direction.  it is currently assumed these will be
            adjacent to one another.  In junctions, diverging lanes
            should not be included."""
            raise NotImplementedError()

        @property
        def lane_to_left(self) -> Tuple[RoadMap.Lane, bool]:
            """Note: left is defined as 90 degrees clockwise relative to the lane heading.
            (I.e., positive `t` in the RefLine coordinate system.)
            Second result is True if lane is in the same direction as this one
            In junctions, diverging lanes should not be included."""
            raise NotImplementedError()

        @property
        def lane_to_right(self) -> Tuple[RoadMap.Lane, bool]:
            """Note: right is defined as 90 degrees counter-clockwise relative to the lane heading.
            (I.e., negative `t` in the RefLine coordinate system.)
            Second result is True if lane is in the same direction as this one.
            In junctions, diverging lanes should not be included."""
            raise NotImplementedError()

        @property
        def incoming_lanes(self) -> List[RoadMap.Lane]:
            """Lanes leading into this lane."""
            raise NotImplementedError()

        @property
        def outgoing_lanes(self) -> List[RoadMap.Lane]:
            """Lanes leading out of this lane."""
            raise NotImplementedError()

        def oncoming_lanes_at_offset(self, offset: float) -> List[RoadMap.Lane]:
            """Returns a list of nearby lanes at offset that are (roughly)
            parallel to this one but go in the opposite direction."""
            raise NotImplementedError()

        @property
        def foes(self) -> List[RoadMap.Lane]:
            """All lanes that in some way intersect with (cross) this one,
            including those that have the same outgoing lane as this one,
            and so might require right-of-way rules.  This should only
            ever happen in junctions."""
            raise NotImplementedError()

        def waypoint_paths_for_pose(
            self, pose: Pose, lookahead: int, route: RoadMap.Route = None
        ) -> List[List[Waypoint]]:
            """Computes equally-spaced Waypoints for all lane paths
            up to lookahead waypoints ahead, starting in this lane at pose.
            Constrains paths to the supplied route if specified."""
            raise NotImplementedError()

        def waypoint_paths_at_offset(
            self, offset: float, lookahead: int = 30, route: RoadMap.Route = None
        ) -> List[List[Waypoint]]:
            """Computes equally-spaced Waypoints for all lane paths
            up to lookahead waypoints ahead, starting offset into this lane.
            Constrains paths to the supplied route if specified."""
            raise NotImplementedError()

        def offset_along_lane(self, world_point: Point) -> float:
            """Get the offset of the given point imposed on this lane."""
            raise NotImplementedError()

        def width_at_offset(self, offset: float) -> float:
            """Get the width of the lane at the given offset."""
            raise NotImplementedError()

        def project_along(
            self, start_offset: float, distance: float
        ) -> Set[Tuple[RoadMap.Lane, float]]:
            """Starting at start_offset along the lane, project locations (lane, offset tuples)
            reachable within distance, not including lane changes."""
            result = set()
            path_stack = {(self, self.length - start_offset)}
            for lane in self.lanes_in_same_direction:
                path_stack.add((lane, lane.length - start_offset))
            while len(path_stack):
                new_stack = set()
                for lane, dist in path_stack:
                    if dist > distance:
                        offset = lane.length + (distance - dist)
                        result.add((lane, offset))
                        continue
                    for out_lane in lane.outgoing_lanes:
                        new_stack.add((out_lane, dist + out_lane.length))
                        for adj_lane in out_lane.lanes_in_same_direction:
                            new_stack.add((adj_lane, dist + adj_lane.length))
                path_stack = new_stack
            return result

        def from_lane_coord(self, lane_point: RefLinePoint) -> Point:
            """Get a world point on the lane from the given lane coordinate point."""
            raise NotImplementedError()

        ## The next 6 methods are "reference" implementations for convenience.
        ## Derived classes may want to extend as well as add a cache.

        def to_lane_coord(self, world_point: Point) -> RefLinePoint:
            """Convert from the given world coordinate to a lane coordinate point."""
            s = self.offset_along_lane(world_point)
            vector = self.vector_at_offset(s)
            normal = np.array([-vector[1], vector[0], 0])
            center_at_s = self.from_lane_coord(RefLinePoint(s=s))
            offcenter_vector = np.array(world_point) - center_at_s
            t_sign = np.sign(np.dot(offcenter_vector, normal))
            t = np.linalg.norm(offcenter_vector) * t_sign
            return RefLinePoint(s=s, t=t)

        def center_at_point(self, point: Point) -> Point:
            """Get the 'center' of the lane closest to the given world coordinate."""
            offset = self.offset_along_lane(point)
            return self.from_lane_coord(RefLinePoint(s=offset))

        def edges_at_point(self, point: Point) -> Tuple[Point, Point]:
            """Get the boundary points perpendicular to the center of the lane closest to the given
             world coordinate.
            Args:
                point:
                    A world coordinate point.
            Returns:
                A pair of points indicating the left boundary and right boundary of the lane.
            """
            offset = self.offset_along_lane(point)
            width = self.width_at_offset(offset)
            left_edge = RefLinePoint(s=offset, t=width / 2)
            right_edge = RefLinePoint(s=offset, t=-width / 2)
            return self.from_lane_coord(left_edge), self.from_lane_coord(right_edge)

        def vector_at_offset(self, start_offset: float) -> np.ndarray:
            """The lane direction vector at the given offset."""
            if start_offset >= self.length:
                s_offset = self.length - 1
                end_offset = self.length
            else:
                s_offset = start_offset
                end_offset = start_offset + 1  # a little further down the lane
            s_offset = max(s_offset, 0)
            p1 = self.from_lane_coord(RefLinePoint(s=s_offset))
            p2 = self.from_lane_coord(RefLinePoint(s=end_offset))
            return np.array(p2) - np.array(p1)

        def center_pose_at_point(self, point: Point) -> Pose:
            """The pose at the center of the lane closest to the given point."""
            offset = self.offset_along_lane(point)
            position = self.from_lane_coord(RefLinePoint(s=offset))
            desired_vector = self.vector_at_offset(offset)
            orientation = fast_quaternion_from_angle(vec_to_radians(desired_vector[:2]))
            return Pose(position=position, orientation=orientation)

        def curvature_radius_at_offset(
            self, offset: float, lookahead: int = 5
        ) -> float:
            """lookahead (in meters) is the size of the window to use
            to compute the curvature, which must be at least 1 to make sense.
            This may return math.inf if the lane is straight."""
            assert lookahead > 1
            if offset + lookahead > self.length:
                return math.inf
            prev_heading_rad = None
            heading_deltas = 0.0
            for i in range(lookahead + 1):
                vec = self.vector_at_offset(offset + i)[:2]
                heading_rad = vec_to_radians(vec[:2])
                if prev_heading_rad is not None:
                    heading_deltas += min_angles_difference_signed(
                        heading_rad, prev_heading_rad
                    )
                prev_heading_rad = heading_rad
            return lookahead / heading_deltas if heading_deltas else math.inf

    class Road(Surface):
        """This is akin to a 'road segment' in real life.
        Many of these might correspond to a single named road in reality."""

        @property
        def road_id(self) -> str:
            """The identifier for this road."""
            raise NotImplementedError()

        @property
        def type(self) -> int:
            """The type of this road."""
            raise NotImplementedError()

        @property
        def type_as_str(self) -> str:
            """The type of this road."""
            raise NotImplementedError()

        @property
        def composite_road(self) -> RoadMap.Road:
            """Return an abstract Road composed of one or more RoadMap.Road segments
            that has been inferred to correspond to one continuous real-world road.
            May return same object as self."""
            return self

        @property
        def is_composite(self) -> bool:
            """Returns True if this Road object was inferred
            and composed out of subordinate Road objects."""
            return False

        @property
        def is_junction(self) -> bool:
            """Note that a junction can be an intersection ('+') or a 'T', 'Y', 'L', etc."""
            raise NotImplementedError()

        @property
        def length(self) -> float:
            """The length of this road."""
            raise NotImplementedError()

        @property
        def incoming_roads(self) -> List[RoadMap.Road]:
            """All roads that lead into this road."""
            raise NotImplementedError()

        @property
        def outgoing_roads(self) -> List[RoadMap.Road]:
            """All roads that lead out of this road."""
            raise NotImplementedError()

        def oncoming_roads_at_point(self, point: Point) -> List[RoadMap.Road]:
            """Returns a list of nearby roads to point that are (roughly)
            parallel to this one but have lanes that go in the opposite direction."""
            raise NotImplementedError()

        @property
        def parallel_roads(self) -> List[RoadMap.Road]:
            """Returns roads that start and end at the same
            point as this one."""
            raise NotImplementedError()

        @property
        def lanes(self) -> List[RoadMap.Lane]:
            """The lanes contained in this road."""
            raise NotImplementedError()

        def lane_at_index(self, index: int) -> RoadMap.Lane:
            """Gets the lane with the given index."""
            raise NotImplementedError()

        def edges_at_point(self, point: Point) -> Tuple[Point, Point]:
            """Get the boundary points perpendicular to the center of the road closest to the given
             world coordinate.
            Args:
                point:
                    A world coordinate point.
            Returns:
                A pair of points indicating the left boundary and right boundary of the road.
            """
            raise NotImplementedError()

    class Feature:
        """Describes a map feature."""

        @property
        def feature_id(self) -> str:
            """The identifier for this feature."""
            raise NotImplementedError()

        @property
        def type(self) -> int:
            """The type of this feature."""
            raise NotImplementedError()

        @property
        def type_as_str(self) -> str:
            """The type of this feature."""
            raise NotImplementedError()

        @property
        def geometry(self) -> List[Point]:
            """The geometry that represents this feature."""
            raise NotImplementedError()

    class Route:
        """Describes a route between two roads."""

        @property
        def roads(self) -> List[RoadMap.Road]:
            """A possibly-unordered list of roads that this route covers"""
            return []

        @property
        def road_length(self) -> float:
            """The total road length of this route."""
            return 0

        @property
        def geometry(self) -> Sequence[Sequence[Tuple[float, float]]]:
            """A sequence of polygon vertices describing the shape of each road on the route"""
            return []

        def distance_between(self, start: Point, end: Point) -> float:
            """Distance along route between two points."""
            raise NotImplementedError()

        def project_along(
            self, start: Point, distance: float
        ) -> Set[Tuple[RoadMap.Lane, float]]:
            """Starting at point on the route, returns a set of possible
            locations (lane and offset pairs) further along the route that
            are distance away, not including lane changes."""
            raise NotImplementedError()


@dataclass(frozen=True)
class Waypoint:
    """Dynamic, based on map and vehicle.  Waypoints start abreast of
    (or near) a vehicle's present location in the nearest Lane and
    are evenly spaced.  These are returned through a vehicle's sensors."""

    # XXX: consider renaming lane_id, lane_index, lane_width
    #      to nearest_lane_id, nearest_lane_index, nearest_lane_width
    pos: np.ndarray  # Point positioned on center of lane
    heading: Heading  # Heading angle of lane at this point (radians)
    lane_id: str  # ID of lane under waypoint
    lane_width: float  # Width of lane at this point (meters)
    speed_limit: float  # Lane speed in m/s
    lane_index: int  # Index of the lane this waypoint is over. 0 is the outer(right) most lane

    def __eq__(self, other) -> bool:
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
