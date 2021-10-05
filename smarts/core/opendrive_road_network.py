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
from cached_property import cached_property
from typing import List, Sequence, Tuple
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
            self._predecessor_elem = self._road_elem.link.predecessor
            self._successor_elem = self._road_elem.link.successor
            self._map = road_map

        @property
        def get_predecessor(self):
            return self._predecessor_elem

        @property
        def get_successor(self):
            return self._successor_elem

        @cached_property
        def is_junction(self) -> bool:
            if self._road_elem.junction:
                return True
            return False

        @cached_property
        def length(self) -> float:
            return self._road_elem._length

        @property
        def road_id(self) -> str:
            return self._road_id

        @cached_property
        def incoming_roads(self) -> List[RoadMap.Road]:
            in_roads = []
            if self._predecessor_elem:
                if self._predecessor_elem.elementType == "road":
                    in_roads.append(
                        self._map.road_by_id(str(self._predecessor_elem.element_id))
                    )
            return in_roads

        @cached_property
        def outgoing_roads(self) -> List[RoadMap.Road]:
            og_roads = []
            if self._successor_elem:
                if self._successor_elem.elementType == "road":
                    og_roads.append(
                        self._map.road_by_id(str(self._successor_elem.element_id))
                    )
            return og_roads

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

        @property
        def lane_id(self) -> str:
            return self._lane_id

        @property
        def road(self) -> RoadMap.Road:
            return self._road

        @property
        def in_junction(self) -> bool:
            return self._road.is_junction

        @cached_property
        def index(self) -> int:
            return self._lane_elem.id

        @cached_property
        def incoming_lanes(self) -> List[RoadMap.Lane]:
            il = []
            lane_link = self._lane_elem.link
            if lane_link.predecessorId:
                road_predecessor = self._road.get_predecessor
                if road_predecessor:
                    if road_predecessor.elementType == "road":
                        pred_lane_id = (
                            str(road_predecessor.element_id)
                            + "_"
                            + str(lane_link.predecessorId)
                        )
                        il.append(self._map.lane_by_id(pred_lane_id))
            return il

        @cached_property
        def outgoing_lanes(self) -> List[RoadMap.Lane]:
            ol = []
            lane_link = self._lane_elem.link
            if lane_link.successorId:
                road_successor = self._road.get_successor
                if road_successor:
                    if road_successor.elementType == "road":
                        succ_lane_id = (
                            str(road_successor.element_id)
                            + "_"
                            + str(lane_link.successorId)
                        )
                        ol.append(self._map.lane_by_id(succ_lane_id))
            return ol

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
        lane = OpenDriveRoadNetwork.Lane(lane_id, lane_elem, self)
        self._lanes[lane_id] = lane
        return lane
