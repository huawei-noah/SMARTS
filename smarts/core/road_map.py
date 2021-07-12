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

from typing import NamedTuple, List, Sequence, Tuple

import numpy as np
from shapely.geometry import Polygon

from .coordinates import BoundingBox, Point, Pose, RefLinePoint
from .utils.math import fast_quaternion_from_angle, vec_to_radians


# TODO:
# - also consider Esri, QGIS and Google Maps formats
# - look at how OpenDrive encodes lane direction
#    -https://www.asam.net/index.php?eID=dumpFile&t=f&f=4089&token=deea5d707e2d0edeeb4fccd544a973de4bc46a09


class RoadMap:
    """Abstract base class from which map implementation classes extend."""

    @property
    def source(self) -> str:
        raise NotImplementedError()

    @property
    def bounding_box(self) -> BoundingBox:
        # maps are assumed to start at the origin
        raise NotImplementedError()

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
        # May be costly.
        raise NotImplementedError()

    def random_route(self, max_route_len: int = 10) -> RoadMap.Route:
        raise NotImplementedError()

    class Lane:
        @property
        def lane_id(self) -> str:
            raise NotImplementedError()

        @property
        def road(self) -> RoadMap.Road:
            raise NotImplementedError()

        @property
        def index(self) -> int:
            """ 0 is outer / right-most (relative to lane heading) lane on road. """
            raise NotImplementedError()

        @property
        def lane_to_left(self) -> Tuple[RoadMap.Lane, bool]:
            """Note: left is defined as 90 degrees clockwise relative to the lane heading.
            Second result is True if lane is in the same direction as this one.
            May return None for lanes in junctions."""
            raise NotImplementedError()

        @property
        def lane_to_right(self) -> Tuple[RoadMap.Lane, bool]:
            """Note: right is defined as 90 degrees counter-clockwise relative to the lane heading.
            Second result is True if lane is in the same direction as this one.
            May return None for lanes in junctions."""
            raise NotImplementedError()

        @property
        def oncoming_lanes(self) -> List[RoadMap.Lane]:
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
        def incoming_lanes(self) -> List[RoadMap.Lane]:
            raise NotImplementedError()

        @property
        def outgoing_lanes(self) -> List[RoadMap.Lane]:
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

        @property
        def outgoing_foes(self) -> Tuple[List[RoadMap.Lane], bool]:
            """All lanes that have the same outgoing lane as this one.
            If right-of-way rules are known, these will be in order of priority,
            otherwise they will be in random order.  The boolean flag
            that is returned will be True if the ordering is by priority."""
            result = [
                incoming
                for incoming in outgoing.incoming_lanes
                for outgoing in self.outgoing_lanes
            ]
            return (list(set(result)), False)

        def to_lane_coord(self, world_point: Point) -> RefLinePoint:
            s = self.offset_along_lane(point)
            vector = self.vector_at_offset(s)
            normal = np.array([-vector[1], vector[0]])
            center_at_s = self.from_lane_coord(RefLinePoint(s=s))
            offcenter_vector = np.array(world_point) - center_at_s
            t_sign = np.sign(np.dot(offcenter_vector, normal))
            t = np.linalg.norm(offcenter_vector) * t_sign
            return RefLinePoint(s=u, t=t)

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
            # assert end_offset <= self.length + add_offset, f"Offset={end_offset} goes out further than the end of lane=({self.lane_id}, length={length})"
            p1 = self.from_lane_coord(RefLinePoint(s=start_offset))
            p2 = self.from_lane_coord(RefLinePoint(s=end_offset))
            return np.array(p2) - np.array(p1)

        def target_pose_at_point(self, point: Point) -> Pose:
            offset = self.offset_along_lane(point)
            position = self.from_lane_coord(RefLinePoint(s=offset))
            desired_vector = self.vector_at_offset(offset)
            orientation = fast_quaternion_from_angle(vec_to_radians(desired_vector[:2]))
            return Pose(position=position, orientation=orientation)

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
            raise NotImplementedError()

        @property
        def length(self) -> int:
            raise NotImplementedError()

        @property
        def incoming_roads(self) -> List[RoadMap.Road]:
            raise NotImplementedError()

        @property
        def outgoing_roads(self) -> List[RoadMap.Road]:
            raise NotImplementedError()

        @property
        def oncoming_roads(self) -> List[RoadMap.Road]:
            raise NotImplementedError()

        @property
        def parallel_roads(self) -> List[RoadMap.Road]:
            raise NotImplementedError()

        @property
        def lanes(self) -> List[RoadMap.Lane]:
            raise NotImplementedError()

        def lane_at_index(self, index: int, direction: bool = True) -> RoadMap.Lane:
            raise NotImplementedError()

        def lanes_by_direction(self, direction: bool) -> List[RoadMap.Lane]:
            """Lanes returned in order of lane index (right-to-left) for a direction.
            direction is arbitrary indicator:
            all True lanes go in the same direction, as do all False lanes,
            but True and False lanes go in opposing directions."""
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

    def create_plan_for_mission(self, mission, find_route: bool = True):  # -> Plan
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
