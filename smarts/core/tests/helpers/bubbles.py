# MIT License
#
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
from collections import defaultdict, namedtuple

from shapely.geometry import LineString
from shapely.ops import split

# TODO: Move these into SMARTS and reuse from tests
BubbleGeometry = namedtuple(
    "BubbleGeometry",
    ["bubble", "airlock_entry", "airlock_exit", "airlock"],
)


def bubble_geometry(bubble, road_network):
    bubble_geometry_ = bubble.zone.to_geometry(road_network)
    airlock_geometry = bubble_geometry_.buffer(bubble.margin)
    split_x, split_y = airlock_geometry.centroid.coords[0]
    divider = LineString([(split_x, -999), (split_x, split_y + 999)])
    airlock_entry_geometry, airlock_exit_geometry = split(airlock_geometry, divider)
    return BubbleGeometry(
        bubble=bubble_geometry_,
        airlock_entry=airlock_entry_geometry,
        airlock_exit=airlock_exit_geometry,
        airlock=airlock_geometry,
    )
