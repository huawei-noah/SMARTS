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

from dataclasses import dataclass
import math
from typing import NamedTuple, List, Sequence, Tuple

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
# - look at how OpenDrive encodes lane direction
#    -https://www.asam.net/index.php?eID=dumpFile&t=f&f=4089&token=deea5d707e2d0edeeb4fccd544a973de4bc46a09


class RoadMap:
    """Base class from which map implementation classes extend."""

    @property
    def source(self) -> str:
        raise NotImplementedError()

    @property
    def bounding_box(self) -> BoundingBox:
        # maps are assumed to start at the origin
        raise NotImplementedError()

    @property
    def xy_offset(self) -> Tuple[float, float]:
        """optional: this is the amount that external coordinates must
        be shifted in order to have the coordinate system (bounding_box)
        start at the origin.  Will be (0. 0) for most maps."""
        # TAI:  get rid of this, fix traffic_history_provider instead
        return (0, 0)

    @property
    def scale_factor(self) -> float:
        # map units per meter
        return 1.0

    def to_glb(self, at_path):
        """ build a glb file for camera rendering and envision """
        raise NotImplementedError()

    def lane_by_id(self, lane_id: str) -> RoadMap.Lane:
        raise NotImplementedError()

    def road_by_id(self, road_id: str) -> RoadMap.Road:
        raise NotImplementedError()

    def nearest_lanes(
        self, point: Point, radius: float = None, include_junctions=True
    ) -> List[Tuple[RoadMap.Lane, float]]:
        raise NotImplementedError()

    def nearest_lane(
        self, point: Point, radius: float = None, include_junctions=True
    ) -> RoadMap.Lane:
        nearest_lanes = self.nearest_lanes(point, radius, include_junctions)
        return nearest_lanes[0][0] if nearest_lanes else None

    def road_with_point(self, point: Point) -> RoadMap.Road:
        raise NotImplementedError()

    def generate_routes(
        self,
        start_road: RoadMap.Road,
        end_road: RoadMap.Road,
        via: Sequence[RoadMap.Road] = None,
        max_to_gen: int = 1,
    ) -> List[RoadMap.Route]:
        """ Routes will be returned in order of increasing road length """
        raise NotImplementedError()

    def random_route(self, max_route_len: int = 10) -> RoadMap.Route:
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

    class Lane:
        @property
        def lane_id(self) -> str:
            raise NotImplementedError()

        @property
        def road(self) -> RoadMap.Road:
            raise NotImplementedError()

        @property
        def speed_limit(self) -> float:
            raise NotImplementedError()

        @property
        def length(self) -> float:
            raise NotImplementedError()

        @property
        def in_junction(self) -> bool:
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
            (I.e., positive `t` in the RefLane coordinate system.)
            Second result is True if lane is in the same direction as this one
            In junctions, diverging lanes should not be included."""
            raise NotImplementedError()

        @property
        def lane_to_right(self) -> Tuple[RoadMap.Lane, bool]:
            """Note: right is defined as 90 degrees counter-clockwise relative to the lane heading.
            (I.e., negative `t` in the RefLane coordinate system.)
            Second result is True if lane is in the same direction as this one.
            In junctions, diverging lanes should not be included."""
            raise NotImplementedError()

        @property
        def incoming_lanes(self) -> List[RoadMap.Lane]:
            raise NotImplementedError()

        @property
        def outgoing_lanes(self) -> List[RoadMap.Lane]:
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

        @property
        def features(self) -> List[RoadMap.Feature]:
            raise NotImplementedError()

        def features_near(self, pose: Pose, radius: float) -> List[RoadMap.Feature]:
            raise NotImplementedError()

        def buffered_shape(self, width: float = 1.0) -> Polygon:
            raise NotImplementedError()

        def point_in_lane(self, point: Point) -> bool:
            raise NotImplementedError()

        def offset_along_lane(self, world_point: Point) -> float:
            raise NotImplementedError()

        def width_at_offset(self, offset: float) -> float:
            raise NotImplementedError()

        def from_lane_coord(self, lane_point: RefLinePoint) -> Point:
            raise NotImplementedError()

        ## The next 6 methods are "reference" implementations for convenience.
        ## Derived classes may want to extend as well as add a cache.

        def to_lane_coord(self, world_point: Point) -> RefLinePoint:
            s = self.offset_along_lane(world_point)
            vector = self.vector_at_offset(s)
            normal = np.array([-vector[1], vector[0], 0])
            center_at_s = self.from_lane_coord(RefLinePoint(s=s))
            offcenter_vector = np.array(world_point) - center_at_s
            t_sign = np.sign(np.dot(offcenter_vector, normal))
            t = np.linalg.norm(offcenter_vector) * t_sign
            return RefLinePoint(s=s, t=t)

        def center_at_point(self, point: Point) -> Point:
            offset = self.offset_along_lane(point)
            return self.from_lane_coord(RefLinePoint(s=offset))

        def edges_at_point(self, point: Point) -> Tuple[Point, Point]:
            offset = self.offset_along_lane(point)
            width = self.width_at_offset(offset)
            left_edge = RefLanePoint(s=offset, t=width / 2)
            right_edge = RefLanePoint(s=offset, t=-width / 2)
            return (self.from_lane_coord(left_edge), self.from_lane_coord(right_edge))

        def vector_at_offset(self, start_offset: float) -> np.ndarray:
            add_offset = 1  # a little further down the lane
            end_offset = start_offset + add_offset
            p1 = self.from_lane_coord(RefLinePoint(s=start_offset))
            p2 = self.from_lane_coord(RefLinePoint(s=end_offset))
            return np.array(p2) - np.array(p1)

        def center_pose_at_point(self, point: Point) -> Pose:
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
            prev_heading_vec = None
            heading_deltas = 0.0
            for i in range(lookahead):
                heading_vec = vec_to_radians(self.vector_at_offset(offset + i))
                if prev_heading_vec:
                    heading_deltas += min_angles_difference_signed(
                        heading_vec, prev_heading_vec
                    )
                prev_heading_vec = heading_vec
            return lookahead / heading_deltas if heading_deltas else math.inf

    class Road:
        """This is akin to a 'road segment' in real life.
        Many of these might correspond to a single named road in reality."""

        @property
        def road_id(self) -> str:
            raise NotImplementedError()

        @property
        def type(self) -> int:
            raise NotImplementedError()

        @property
        def type_as_str(self) -> str:
            raise NotImplementedError()

        @property
        def is_junction(self) -> bool:
            """Note that a junction can be an intersection ('+') or a 'T', 'Y', 'L', etc."""
            raise NotImplementedError()

        @property
        def length(self) -> float:
            raise NotImplementedError()

        @property
        def incoming_roads(self) -> List[RoadMap.Road]:
            raise NotImplementedError()

        @property
        def outgoing_roads(self) -> List[RoadMap.Road]:
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
            raise NotImplementedError()

        def lane_at_index(self, index: int) -> RoadMap.Lane:
            raise NotImplementedError()

        def point_on_road(self, point: Point) -> bool:
            raise NotImplementedError()

        def edges_at_point(self, point: Point) -> Tuple[Point, Point]:
            raise NotImplementedError()

        def buffered_shape(self, width: float = 1.0) -> Polygon:
            raise NotImplementedError()

    class Feature:
        @property
        def feature_id(self) -> str:
            raise NotImplementedError()

        @property
        def type(self) -> int:
            raise NotImplementedError()

        @property
        def type_as_str(self) -> str:
            raise NotImplementedError()

        @property
        def geometry(self) -> List[Point]:
            raise NotImplementedError()

    class Route:
        @property
        def roads(self) -> List[RoadMap.Road]:
            """An (unordered) list of roads that this route covers"""
            return []

        @property
        def road_length(self) -> float:
            return 0

        @property
        def geometry(self) -> Sequence[Sequence[Tuple[float, float]]]:
            """A sequence of polygons describing the shape of each road on the route"""
            return []

        @property
        def crossing_lanes(self) -> List[RoadMap.Lane]:
            raise NotImplementedError()

        def distance_between(self, start: Point, end: Point) -> float:
            """ Distance along route between two points.  """
            raise NotImplementedError()


@dataclass(frozen=True)
class Waypoint:
    """Dynamic, based on map and vehicle.  Waypoints
    start abreast of a vehicle's present location in the nearest Lane
    and are then interpolated such that they're evenly spaced.
    These are returned through a vehicle's sensors."""

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
