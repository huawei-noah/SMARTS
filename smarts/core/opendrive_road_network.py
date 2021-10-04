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
import logging

from lxml import etree
from opendrive2lanelet.opendriveparser.elements.opendrive import OpenDrive
from opendrive2lanelet.opendriveparser.elements.road import Road as RoadElement
from opendrive2lanelet.opendriveparser.elements.roadLanes import Lane as LaneElement
from opendrive2lanelet.opendriveparser.parser import parse_opendrive

from smarts.core.road_map import RoadMap


class OpenDriveRoadNetwork(RoadMap):
    def __init__(self, network: OpenDrive, xodr_file: str):
        self._log = logging.getLogger(self.__class__.__name__)
        self._network = network
        self._xodr_file = xodr_file
        self._lanes = {}
        self._roads = {}
        self._lanepoints = None

    @classmethod
    def from_file(
        cls,
        xodr_file,
    ):
        with open(xodr_file, "r") as f:
            network = parse_opendrive(etree.parse(f).getroot())

        return cls(
            network,
            xodr_file,
        )

    @property
    def source(self) -> str:
        """ This is the .xodr file of the OpenDRIVE map. """
        return self._xodr_file

    class Road(RoadMap.Road):
        def __init__(self, road_id: str, road_elem: RoadElement, road_map: RoadMap):
            self._road_id = road_id
            self._road_elem = road_elem
            self._map = road_map

    def road_by_id(self, road_id: str) -> RoadMap.Road:
        road = self._roads.get(road_id)
        if road:
            return road
        road_elem = self._network.getRoad(int(road_id))
        if not road_elem:
            self._log.warning(
                f"OpenDriveRoadNetwork got request for unknown road_id '{road_id}'"
            )
            return None
        road = OpenDriveRoadNetwork.Road(road_id, road_elem, self)
        self._roads[road_id] = road
        return road

    class Lane(RoadMap.Lane):
        def __init__(self, lane_id: str, lane_elem: LaneElement, road_map):
            self._lane_id = lane_id
            self._lane_elem = lane_elem
            self._map = road_map
            self._road = road_map.road_by_id(lane_elem.parentRoad.id)
            assert self._road

    def lane_by_id(self, lane_id: str) -> RoadMap.Lane:
        lane = self._lanes.get(lane_id)
        if lane:
            return lane
        lane_elem = None
        split_lst = lane_id.split("_")
        road_id, od_lane_id = split_lst[0], split_lst[1]

        road_elem = self._network.getRoad(int(road_id))
        if not road_elem:
            self._log.warning(
                f"OpenDriveRoadNetwork got request for unknown road_id '{road_id}'"
            )
            return None

        for lane_section in road_elem.lanes.lane_sections:
            for od_lane in lane_section.allLanes:
                if od_lane.id == int(od_lane_id):
                    lane_elem = od_lane
                    break
            if lane_elem:
                break
        if not lane_elem:
            self._log.warning(
                f"OpenDriveRoadNetwork got request for unknown lane_id '{lane_id}'"
            )
            return None
        lane = OpenDriveRoadNetwork.Road(lane_id, lane_elem, self)
        self._lanes[lane_id] = lane
        return lane
