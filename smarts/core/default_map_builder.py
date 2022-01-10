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
from dataclasses import replace
from typing import NamedTuple, Tuple
from pathlib import Path
from smarts.core.road_map import RoadMap
from smarts.core.utils.file import file_md5_hash

_existing_map = None


# This function should be re-callable (although caching is up to the implementation).
# The idea here is that anything in SMARTS that needs to use a RoadMap
# can call this builder to get or create one of default type.
#
# Downstream developers who want to extend SMARTS to support other
# map formats (by extending the RoadMap base class) can create their
# own version of this to reference from a MapSpec within their
# scenario folder(s) and shouldn't have to change much else.

supported_maps = [
    "*.net.xml",  # SUMO
    "*.xodr",  # OpenDRIVE
]


def get_road_map(map_spec) -> Tuple[RoadMap, str]:
    """@return a RoadMap object and a hash
    that uniquely identifies it. Changes to the hash
    should signify that the map is different enough
    that map-related caches should be reloaded.
    If possible, the RoadMap object may be cached here
    and re-used.
    """
    assert map_spec, "A road map spec must be specified"
    assert map_spec.source, "A road map source must be specified"

    global _existing_map
    if _existing_map:
        if _existing_map.obj.is_same_map(map_spec):
            return _existing_map.obj, _existing_map.map_hash
        import gc

        # Try to only keep one map around at a time...
        del _existing_map
        _existing_map = None
        gc.collect()

    if not os.path.isfile(map_spec.source):
        scenario_root = Path(map_spec.source)
        path_found = False
        for i, map_name in enumerate(supported_maps):
            map_paths = list(scenario_root.rglob(map_name))
            if len(map_paths) == 1:
                map_spec = replace(map_spec, source=str(map_paths[0]))
                path_found = True
                break
            elif len(map_paths) > 1:
                for map_path in map_paths:
                    if not str(map_path).endswith("AUTOGEN.net.xml"):
                        map_spec = replace(map_spec, source=str(map_path))
                        path_found = True
                        break
                break
            else:
                continue

        if not path_found:
            raise FileNotFoundError(
                f"Unable to find map in map_source={map_spec.source}."
            )

    road_map = None
    if map_spec.source.endswith(".net.xml"):
        # Keep this a conditional import so Sumo does not have to be
        # imported if not necessary:
        from smarts.core.sumo_road_network import SumoRoadNetwork

        road_map = SumoRoadNetwork.from_spec(map_spec)

    elif map_spec.source.endswith(".xodr"):
        from smarts.core.opendrive_road_network import OpenDriveRoadNetwork

        road_map = OpenDriveRoadNetwork.from_spec(map_spec)

    road_map_hash = file_md5_hash(road_map.source)

    from smarts.sstudio.types import MapSpec

    class _RoadMapInfo(NamedTuple):
        map_spec: MapSpec
        obj: RoadMap
        map_hash: str

    _existing_map = _RoadMapInfo(map_spec, road_map, road_map_hash)

    return road_map, road_map_hash
