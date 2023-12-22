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


# A MapBuilder should return an object derived from the RoadMap base class
# and a hash that uniquely identifies it (changes to the hash should signify
# that the map is different enough that map-related caches should be reloaded).
#
# This function should be re-callable (although caching is up to the implementation).
# The idea here is that anything in SMARTS that needs to use a RoadMap
# can call this builder to get or create one as necessary.
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

from smarts.core.default_map_builder import get_road_map
from smarts.core.road_map import RoadMap

MapBuilder = Callable[[Any], Tuple[Optional[RoadMap], Optional[str]]]


@dataclass(frozen=True)
class MapSpec:
    """A map specification that describes how to generate a roadmap."""

    source: str
    """A path or URL or name uniquely designating the map source."""
    lanepoint_spacing: float = 1.0
    """The default distance between pre-generated Lane Points (Waypoints)."""
    default_lane_width: Optional[float] = None
    """If specified, the default width (in meters) of lanes on this map."""
    shift_to_origin: bool = False
    """If True, upon creation a map whose bounding-box does not intersect with
    the origin point (0,0) will be shifted such that it does."""
    builder_fn: MapBuilder = get_road_map
    """If specified, this should return an object derived from the RoadMap base class
    and a hash that uniquely identifies it (changes to the hash should signify
    that the map is different enough that map-related caches should be reloaded).
    The parameter is this MapSpec object itself.
    If not specified, this currently defaults to a function that creates
    SUMO road networks (get_road_map()) in smarts.core.default_map_builder."""
