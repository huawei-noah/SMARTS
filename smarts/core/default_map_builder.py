# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
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
from __future__ import annotations

import os
import sys
import warnings
from enum import IntEnum
from typing import TYPE_CHECKING, NamedTuple, Optional, Tuple

from smarts.core.utils.file import file_md5_hash, path2hash

if TYPE_CHECKING:
    from smarts.core.road_map import RoadMap
    from smarts.sstudio.sstypes import MapSpec


_existing_map = None


class _RoadMapInfo(NamedTuple):
    map_spec: MapSpec  # pytype: disable=invalid-annotation
    obj: RoadMap
    map_hash: str


def _cache_result(map_spec, road_map: RoadMap, road_map_hash: str):
    global _existing_map
    _existing_map = _RoadMapInfo(map_spec, road_map, road_map_hash)


def _clear_cache():
    global _existing_map
    if _existing_map:
        import gc

        # Try to only keep one map around at a time...
        del _existing_map
        _existing_map = None
        gc.collect()


class MapType(IntEnum):
    """The format of a map."""

    Unknown = 0
    Sumo = 1
    Opendrive = 2
    Waymo = 3
    Argoverse = 4


def find_mapfile_in_dir(map_dir: str) -> Tuple[MapType, str]:
    """Looks in a given directory for a supported map file."""
    map_filename_type = {
        "map.net.xml": MapType.Sumo,
        "shifted_map-AUTOGEN.net.xml": MapType.Sumo,
        "map.xodr": MapType.Opendrive,
    }
    map_type = MapType.Unknown
    map_path = map_dir
    for f in os.listdir(map_dir):
        cand_map_type = map_filename_type.get(f)
        if cand_map_type is not None:
            return cand_map_type, os.path.join(map_dir, f)
        if f.endswith(".net.xml"):
            map_type = MapType.Sumo
            map_path = os.path.join(map_dir, f)
        elif f.endswith(".xodr"):
            map_type = MapType.Opendrive
        elif ".tfrecord" in f:
            map_type = MapType.Waymo
            map_path = os.path.join(map_dir, f)
        elif "log_map_archive" in f:
            map_type = MapType.Argoverse
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
        map_type, map_source = find_mapfile_in_dir(map_spec.source)
    else:
        map_type = MapType.Unknown
        map_source = map_spec.source
        if map_source.endswith(".net.xml"):
            map_type = MapType.Sumo
        elif map_source.endswith(".xodr"):
            map_type = MapType.Opendrive
        elif ".tfrecord" in map_source:
            map_type = MapType.Waymo

    if map_type == MapType.Sumo:
        from smarts.core.sumo_road_network import SumoRoadNetwork

        map_class = SumoRoadNetwork

    elif map_type == MapType.Opendrive:
        from smarts.core.utils.custom_exceptions import OpenDriveException

        try:
            from smarts.core.opendrive_road_network import OpenDriveRoadNetwork
        except ImportError:
            raise OpenDriveException.required_to("use OpenDRIVE maps")
        map_class = OpenDriveRoadNetwork

    elif map_type == MapType.Waymo:
        from smarts.core.waymo_map import WaymoMap

        map_class = WaymoMap
    elif map_type == MapType.Argoverse:
        try:
            from smarts.core.argoverse_map import (
                ArgoverseMap,  # pytype: disable=import-error
            )
        except (ImportError, ModuleNotFoundError):
            print(sys.exc_info())
            print(
                "Missing dependencies for Argoverse. Install them using the command `pip install -e .[argoverse]` at the source directory."
            )
            return None, None
        map_class = ArgoverseMap
    else:
        warnings.warn(
            f"A map source for road surface generation can not be resolved from the given reference: `{map_spec.source}`.",
            category=UserWarning,
        )
        return None, None

    if _existing_map:
        if isinstance(_existing_map.obj, map_class) and _existing_map.obj.is_same_map(
            map_spec
        ):
            return _existing_map.obj, _existing_map.map_hash
        _clear_cache()

    road_map = map_class.from_spec(map_spec)
    if road_map is None:
        return None, None

    if os.path.isfile(road_map.source):
        road_map_hash = file_md5_hash(road_map.source)
    else:
        road_map_hash = path2hash(road_map.source)
    _cache_result(map_spec, road_map, road_map_hash)

    return road_map, road_map_hash
