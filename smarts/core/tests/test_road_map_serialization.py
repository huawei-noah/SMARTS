# MIT License
#
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
import logging
from typing import List

import pytest
from helpers.scenario import maps_dir

from smarts.core.road_map import RoadMap
from smarts.sstudio.types import MapSpec


def waymo_map() -> RoadMap:
    pass


def sumo_map() -> RoadMap:
    from smarts.core.sumo_road_network import SumoRoadNetwork

    map_spec = MapSpec(str(maps_dir()))
    road_network = SumoRoadNetwork.from_spec(map_spec)
    return road_network


def opendrive_map() -> RoadMap:
    pass


@pytest.fixture
def road_maps() -> List[RoadMap]:
    map_funcs = [
        # waymo_map,
        sumo_map,
        # opendrive_map,
    ]
    yield (m() for m in map_funcs)


def test_map_serializations(road_maps: List[RoadMap]):
    for m in road_maps:
        m: RoadMap = m
        logging.getLogger().info("Map source: `%s`", m.source)

        # Test serialization of the map
        srm = RoadMap.serialize(m)
        mrm = RoadMap.deserialize(srm)

        assert m.is_same_map(mrm)
