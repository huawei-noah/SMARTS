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
from pathlib import Path
from typing import NamedTuple, Optional, Tuple

from smarts.core.road_map import RoadMap
from smarts.core.utils.file import file_md5_hash, path2hash

_existing_map = None


def _cache_result(map_spec, road_map, road_map_hash: str):
    global _existing_map
    from smarts.sstudio.types import MapSpec

    class _RoadMapInfo(NamedTuple):
        map_spec: MapSpec
        obj: RoadMap
        map_hash: str

    _existing_map = _RoadMapInfo(map_spec, road_map, road_map_hash)


def _clear_cache():
    global _existing_map
    if _existing_map:
        import gc

        # Try to only keep one map around at a time...
        del _existing_map
        _existing_map = None
        gc.collect()


_UNKNOWN_MAP = 0
_SUMO_MAP = 1
_OPENDRIVE_MAP = 2


def _find_mapfile_in_dir(map_dir: str) -> Tuple[int, str]:
    map_filename_type = {
        "map.net.xml": _SUMO_MAP,
        "shifted_map-AUTOGEN.net.xml": _SUMO_MAP,
        "map.xodr": _OPENDRIVE_MAP,
    }
    map_type = _UNKNOWN_MAP
    map_path = map_dir
    for f in os.listdir(map_dir):
        cand_map_type = map_filename_type.get(f)
        if cand_map_type is not None:
            return cand_map_type, os.path.join(map_dir, f)
        if f.endswith(".net.xml"):
            map_type = _SUMO_MAP
            map_path = os.path.join(map_dir, f)
        elif f.endswith(".xodr"):
            map_type = _OPENDRIVE_MAP
            map_path = os.path.join(map_dir, f)
    return map_type, map_path


# This function should be re-callable (although caching is up to the implementation).
# The idea here is that anything in SMARTS that needs to use a RoadMap
# can call this builder to get or create one of default type.
#
# Downstream developers who want to extend SMARTS to support other
# map formats (by extending the RoadMap base class) can create their
# own version of this to reference from a MapSpec within their
# scenario folder(s) and shouldn't have to change much else.
def get_road_map(map_spec) -> Tuple[Optional[RoadMap], Optional[str]]:
    """@return a RoadMap object and a hash
    that uniquely identifies it. Changes to the hash
    should signify that the map is different enough
    that map-related caches should be reloaded.
    If possible, the RoadMap object may be cached here
    and re-used.
    """
    assert map_spec, "A road map spec must be specified"
    assert map_spec.source, "A road map source must be specified"

    if os.path.isdir(map_spec.source):
        map_type, map_source = _find_mapfile_in_dir(map_spec.source)
    else:
        map_type = _UNKNOWN_MAP
        map_source = map_spec.source
        if map_source.endswith(".net.xml"):
            map_type = _SUMO_MAP
        elif map_source.endswith(".xodr"):
            map_type = _OPENDRIVE_MAP

    if map_type == _SUMO_MAP:
        from smarts.core.sumo_road_network import SumoRoadNetwork

        map_class = SumoRoadNetwork

    elif map_type == _OPENDRIVE_MAP:
        from smarts.core.opendrive_road_network import OpenDriveRoadNetwork

        map_class = OpenDriveRoadNetwork

    else:
        return None, None

    if _existing_map:
        if isinstance(_existing_map.obj, map_class) and _existing_map.obj.is_same_map(
            map_spec
        ):
            return _existing_map.obj, _existing_map.map_hash
        _clear_cache()

    road_map = map_class.from_spec(map_spec)
    if os.path.isfile(road_map.source):
        road_map_hash = file_md5_hash(road_map.source)
    else:
        road_map_hash = path2hash(road_map.source)
    _cache_result(map_spec, road_map, road_map_hash)

    return road_map, road_map_hash
