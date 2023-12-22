# MIT License
#
# Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.
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
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from shapely.affinity import rotate as shapely_rotate
from shapely.affinity import translate as shapely_translate
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiPolygon,
    Point,
    Polygon,
    box,
)
from shapely.ops import split, unary_union

from smarts.core.coordinates import RefLinePoint
from smarts.core.road_map import RoadMap
from smarts.core.utils.core_math import rotate_cw_around_point


@dataclass(frozen=True)
class Zone:
    """The base for a descriptor that defines a capture area."""

    def to_geometry(self, road_map: Optional[RoadMap] = None) -> Polygon:
        """Generates the geometry from this zone."""
        raise NotImplementedError


@dataclass(frozen=True)
class MapZone(Zone):
    """A descriptor that defines a capture area."""

    start: Tuple[str, int, float]
    """The (road_id, lane_index, offset) details of the starting location.

    road_id:
        The starting road by name.
    lane_index:
        The lane index from the rightmost lane.
    offset:
        The offset in meters into the lane. Also acceptable 'max' or 'random'.
    """
    length: float
    """The length of the geometry along the center of the lane. Also acceptable 'max'."""
    n_lanes: int = 2
    """The number of lanes from right to left that this zone covers."""

    def to_geometry(self, road_map: Optional[RoadMap]) -> Polygon:
        """Generates a map zone over a stretch of the given lanes."""

        assert (
            road_map is not None
        ), f"{self.__class__.__name__} requires a road map to resolve geometry."

        def resolve_offset(offset, geometry_length, lane_length):
            if offset == "base":
                return 0
            # push off of end of lane
            elif offset == "max":
                return lane_length - geometry_length
            elif offset == "random":
                return random.uniform(0, lane_length - geometry_length)
            else:
                return float(offset)

        def pick_remaining_shape_after_split(geometry_collection, expected_point):
            lane_shape = geometry_collection
            if not isinstance(lane_shape, GeometryCollection):
                return lane_shape

            # For simplicity, we only deal w/ the == 1 or 2 case
            if len(lane_shape.geoms) not in {1, 2}:
                return None

            if len(lane_shape.geoms) == 1:
                return lane_shape.geoms[0]

            # We assume that there are only two split shapes to choose from
            keep_index = 0
            if lane_shape.geoms[1].minimum_rotated_rectangle.contains(expected_point):
                # 0 is the discard piece, keep the other
                keep_index = 1

            lane_shape = lane_shape.geoms[keep_index]

            return lane_shape

        def split_lane_shape_at_offset(
            lane_shape: Polygon, lane: RoadMap.Lane, offset: float
        ):
            # XXX: generalize to n-dim
            width_2, _ = lane.width_at_offset(offset)
            point = np.array(lane.from_lane_coord(RefLinePoint(offset)))[:2]
            lane_vec = lane.vector_at_offset(offset)[:2]

            perp_vec_right = rotate_cw_around_point(lane_vec, np.pi / 2, origin=(0, 0))
            perp_vec_right = (
                perp_vec_right / max(np.linalg.norm(perp_vec_right), 1e-3) * width_2
                + point
            )

            perp_vec_left = rotate_cw_around_point(lane_vec, -np.pi / 2, origin=(0, 0))
            perp_vec_left = (
                perp_vec_left / max(np.linalg.norm(perp_vec_left), 1e-3) * width_2
                + point
            )

            split_line = LineString([perp_vec_left, perp_vec_right])
            return split(lane_shape, split_line)

        lane_shapes = []
        road_id, lane_idx, offset = self.start
        road = road_map.road_by_id(road_id)
        buffer_from_ends = 1e-6
        for lane_idx in range(lane_idx, lane_idx + self.n_lanes):
            lane = road.lane_at_index(lane_idx)
            lane_length = lane.length
            geom_length = self.length

            if geom_length > lane_length:
                logging.debug(
                    f"Geometry is too long={geom_length} with offset={offset} for "
                    f"lane={lane.lane_id}, using length={lane_length} instead"
                )
                geom_length = lane_length

            assert geom_length > 0  # Geom length is negative

            lane_offset = resolve_offset(offset, geom_length, lane_length)
            lane_offset += buffer_from_ends
            width, _ = lane.width_at_offset(lane_offset)  # TODO
            lane_shape = lane.shape(0.3, width)  # TODO

            geom_length = max(geom_length - buffer_from_ends, buffer_from_ends)
            lane_length = max(lane_length - buffer_from_ends, buffer_from_ends)

            min_cut = min(lane_offset, lane_length)
            # Second cut takes into account shortening of geometry by `min_cut`.
            max_cut = min(min_cut + geom_length, lane_length)

            midpoint = Point(
                *lane.from_lane_coord(RefLinePoint(s=lane_offset + geom_length * 0.5))
            )

            lane_shape = split_lane_shape_at_offset(lane_shape, lane, min_cut)
            lane_shape = pick_remaining_shape_after_split(lane_shape, midpoint)
            if lane_shape is None:
                continue

            lane_shape = split_lane_shape_at_offset(
                lane_shape,
                lane,
                max_cut,
            )
            lane_shape = pick_remaining_shape_after_split(lane_shape, midpoint)
            if lane_shape is None:
                continue

            lane_shapes.append(lane_shape)

        geom = unary_union(MultiPolygon(lane_shapes))
        return geom


@dataclass(frozen=True)
class PositionalZone(Zone):
    """A descriptor that defines a capture area at a specific XY location."""

    # center point
    pos: Tuple[float, float]
    """A (x,y) position of the zone in the scenario."""
    size: Tuple[float, float]
    """The (length, width) dimensions of the zone."""
    rotation: Optional[float] = None
    """The heading direction of the bubble. (radians, clock-wise rotation)"""

    def to_geometry(self, road_map: Optional[RoadMap] = None) -> Polygon:
        """Generates a box zone at the given position."""
        w, h = self.size
        x, y = self.pos[:2]
        p0 = (-w / 2, -h / 2)  # min
        p1 = (w / 2, h / 2)  # max
        poly = Polygon([p0, (p0[0], p1[1]), p1, (p1[0], p0[1])])
        if self.rotation is not None:
            poly = shapely_rotate(poly, self.rotation, use_radians=True)
        return shapely_translate(poly, xoff=x, yoff=y)


@dataclass(frozen=True)
class ConfigurableZone(Zone):
    """A descriptor for a zone with user-defined geometry."""

    ext_coordinates: List[Tuple[float, float]]
    """external coordinates of the polygon
    < 2 points provided: error
    = 2 points provided: generates a box using these two points as diagonal
    > 2 points provided: generates a polygon according to the coordinates"""
    rotation: Optional[float] = None
    """The heading direction of the bubble(radians, clock-wise rotation)"""

    def __post_init__(self):
        if (
            not self.ext_coordinates
            or len(self.ext_coordinates) < 2
            or not isinstance(self.ext_coordinates[0], tuple)
        ):
            raise ValueError(
                "Two points or more are needed to create a polygon. (less than two points are provided)"
            )

        x_set = set(point[0] for point in self.ext_coordinates)
        y_set = set(point[1] for point in self.ext_coordinates)
        if len(x_set) == 1 or len(y_set) == 1:
            raise ValueError(
                "Parallel line cannot form a polygon. (points provided form a parallel line)"
            )

    def to_geometry(self, road_map: Optional[RoadMap] = None) -> Polygon:
        """Generate a polygon according to given coordinates"""
        poly = None
        if (
            len(self.ext_coordinates) == 2
        ):  # if user only specified two points, create a box
            x_min = min(self.ext_coordinates[0][0], self.ext_coordinates[1][0])
            x_max = max(self.ext_coordinates[0][0], self.ext_coordinates[1][0])
            y_min = min(self.ext_coordinates[0][1], self.ext_coordinates[1][1])
            y_max = max(self.ext_coordinates[0][1], self.ext_coordinates[1][1])
            poly = box(x_min, y_min, x_max, y_max)

        else:  # else create a polygon according to the coordinates
            poly = Polygon(self.ext_coordinates)

        if self.rotation is not None:
            poly = shapely_rotate(poly, self.rotation, use_radians=True)
        return poly


@dataclass(frozen=True)
class RoadSurfacePatch:
    """A descriptor that defines a patch of road surface with a different friction coefficient."""

    zone: Zone
    """The zone which to capture vehicles."""
    begin_time: int
    """The start time in seconds of when this surface is active."""
    end_time: int
    """The end time in seconds of when this surface is active."""
    friction_coefficient: float
    """The surface friction coefficient."""
