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

import os
from typing import Tuple
from smarts.core.road_map import RoadMap
from smarts.core.utils.file import file_md5_hash


# The idea here is that anything in SMARTS that needs to use a RoadMap
# can call this factory to create one of default type.
#
# Downstream developers who want to extend SMARTS to support other
# map formats (by extending the RoadMap base class) can replace this
# file with their own version and shouldn't have to change much else.


def create_road_map(
    map_source: str,
    lanepoint_spacing: float = None,
    default_lane_width: float = None,
) -> Tuple[RoadMap, str]:
    """@return a RoadMap object and a hash
    that uniquely identifies it. Changes to the hash
    should signify that the map is different enough
    that map-related caches should be reloaded."""

    assert map_source, "A road map source must be specified"

    map_path = map_source
    if not os.path.isfile(map_path):
        map_path = os.path.join(map_source, "map.net.xml")
        if not os.path.exists(map_path):
            raise Exception(f"Unable to find map in map_source={map_source}.")

    # Keep this a conditional import so Sumo does not have to be
    # imported if not necessary:
    from smarts.core.sumo_road_network import SumoRoadNetwork

    road_map = SumoRoadNetwork.from_file(
        map_path,
        default_lane_width=default_lane_width,
        lanepoint_spacing=lanepoint_spacing,
    )

    road_map_hash = file_md5_hash(road_map.source)

    return road_map, road_map_hash
