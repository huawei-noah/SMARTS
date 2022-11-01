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
import math
from typing import List, Optional, Union

from smarts.core.coordinates import Point
from smarts.core.utils.math import is_close

## Note:  the methods in this file are not efficient / optimized,
## so they shouldn't be used on the critical path(s).


def euclidean_distance(p1: Point, p2: Point) -> float:
    """The distance taking measuring a direct line between p1 and p2."""
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return math.sqrt(dx * dx + dy * dy)


def position_at_offset(p1: Point, p2: Point, offset: float) -> Optional[Point]:
    """A point between p1 and p2 given an offset less than the distance between p1 and p2."""
    if is_close(offset, 0.0):  # for pathological cases with dist == 0 and offset == 0
        return p1

    dist = euclidean_distance(p1, p2)

    if is_close(dist, offset):
        return p2

    if offset > dist:
        return None

    x = p1[0] + (p2[0] - p1[0]) * (offset / dist)
    y = p1[1] + (p2[1] - p1[1]) * (offset / dist)
    return Point(x, y)


def offset_along_shape(point: Point, shape: List[Point]) -> Union[float, int]:
    """An offset on a shape defined as a vector path determined by the closest location on the
    path to the point.
    """
    if point not in shape:
        return polygon_offset_with_minimum_distance_to_point(point, shape)
    offset = 0
    for i in range(len(shape) - 1):
        if shape[i] == point:
            break
        offset += euclidean_distance(shape[i], shape[i + 1])
    return offset


def position_at_shape_offset(shape: List[Point], offset: float) -> Optional[Point]:
    """A point defined as the offset into a shape defined as vector path."""
    seen_length = 0
    curr = shape[0]
    for next_p in shape[1:]:
        next_length = euclidean_distance(curr, next_p)
        if seen_length + next_length > offset:
            return position_at_offset(curr, next_p, offset - seen_length)
        seen_length += next_length
        curr = next_p
    return shape[-1]


def line_offset_with_minimum_distance_to_point(
    point: Point,
    line_start: Point,
    line_end: Point,
    perpendicular: bool = False,
) -> Union[float, int]:
    """Return the offset from line (line_start, line_end) where the distance to
    point is minimal"""
    p = point
    p1 = line_start
    p2 = line_end
    d = euclidean_distance(p1, p2)
    u = ((p[0] - p1[0]) * (p2[0] - p1[0])) + ((p[1] - p1[1]) * (p2[1] - p1[1]))
    if u < 0.0:
        return 0.0
    if d == 0.0 or u > d * d:
        if perpendicular:
            return -1
        return d
    return u / d


def polygon_offset_with_minimum_distance_to_point(
    point: Point, polygon: List[Point]
) -> Union[float, int]:
    """Return the offset and the distance from the polygon start where the distance to the point is minimal"""
    p = point
    s = polygon
    seen = 0
    min_dist = 1e400
    min_offset = -1
    for i in range(len(s) - 1):
        p_offset = line_offset_with_minimum_distance_to_point(p, s[i], s[i + 1])
        dist = (
            min_dist
            if p_offset == -1
            else euclidean_distance(p, position_at_offset(s[i], s[i + 1], p_offset))
        )
        if dist < min_dist:
            min_dist = dist
            min_offset = p_offset + seen
        seen += euclidean_distance(s[i], s[i + 1])
    return min_offset


def distance_point_to_line(
    point: Point,
    line_start: Point,
    line_end: Point,
    perpendicular: bool = False,
) -> Union[float, int]:
    """Return the minimum distance between point and the line (line_start, line_end)"""
    p1 = line_start
    p2 = line_end
    offset = line_offset_with_minimum_distance_to_point(
        point, line_start, line_end, perpendicular
    )
    if offset == -1:
        return -1
    if offset == 0:
        return euclidean_distance(point, p1)
    u = offset / euclidean_distance(line_start, line_end)
    intersection = (p1[0] + u * (p2[0] - p1[0]), p1[1] + u * (p2[1] - p1[1]))
    return euclidean_distance(point, intersection)


def distance_point_to_polygon(
    point: Point, polygon: List[Point], perpendicular: bool = False
) -> Union[float, int]:
    """Return the minimum distance between point and polygon"""
    p = point
    s = polygon
    min_dist = None
    for i in range(len(s) - 1):
        dist = distance_point_to_line(p, s[i], s[i + 1], perpendicular)
        if dist == -1 and perpendicular and i != 0:
            # distance to inner corner
            dist = euclidean_distance(point, s[i])
        if dist != -1:
            if min_dist is None or dist < min_dist:
                min_dist = dist
    if min_dist is not None:
        return min_dist
    return -1
