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

supported_maps = [
    "map.net.xml",  # SUMO
    "map.xodr",  # OpenDRIVE
]


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
        for i, map_name in enumerate(supported_maps):
            map_path = os.path.join(map_source, map_name)
            if os.path.exists(map_path):
                break
            if i == len(supported_maps) - 1:
                raise FileNotFoundError(f"Unable to find map in map_source={map_source}.")

    road_map = None
    if map_path.endswith("map.net.xml"):
        # Keep this a conditional import so Sumo does not have to be
        # imported if not necessary:
        from smarts.core.sumo_road_network import SumoRoadNetwork

        road_map = SumoRoadNetwork.from_file(
            map_path,
            default_lane_width=default_lane_width,
            lanepoint_spacing=lanepoint_spacing,
        )
    elif map_path.endswith("map.xodr"):
        from smarts.core.opendrive_road_network import OpenDriveRoadNetwork

        road_map = OpenDriveRoadNetwork.from_file(
            map_path,
            default_lane_width=default_lane_width,
            lanepoint_spacing=lanepoint_spacing,
        )
    if road_map:
        road_map_hash = file_md5_hash(road_map.source)
    else:
        road_map_hash = None
    return road_map, road_map_hash
