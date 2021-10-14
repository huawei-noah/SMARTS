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
from typing import Dict, List, Sequence, Tuple

import numpy as np
from cached_property import cached_property
from lxml import etree
from numpy.core.defchararray import index
from opendrive2lanelet.opendriveparser.elements.opendrive import \
    OpenDrive as OpenDriveElement
from opendrive2lanelet.opendriveparser.elements.road import Road as RoadElement
from opendrive2lanelet.opendriveparser.elements.roadLanes import \
    Lane as LaneElement
from opendrive2lanelet.opendriveparser.parser import parse_opendrive

from smarts.core.road_map import RoadMap


class OpenDriveRoadNetwork(RoadMap):
    def __init__(self, xodr_file: str):
        self._log = logging.getLogger(self.__class__.__name__)
        self._xodr_file = xodr_file
        self._roads: Dict[str, OpenDriveRoadNetwork.Road] = {}
        self._lanes: Dict[str, OpenDriveRoadNetwork.Lane]  = {}
        self._lanepoints = None
        self._junctions = {}
        # self._junctions = network.junctions
        self._junction_connections = {}
        # self._precompute_junction_connections()

    @classmethod
    def from_file(
        cls,
        xodr_file,
    ):
        map = cls(xodr_file)
        map.load()
        return map

    @staticmethod
    def _elem_id(elem):
        if type(elem) == RoadElement:
            return str(elem.id)
        elif type(elem) == LaneElement:
            return f"{elem.parentRoad.id}_{elem.lane_section.idx}_{elem.id}"
        else:
            return None

    def load(self):
        # Parse the xml definition into an initial representation
        od: OpenDriveElement = None
        with open(self._xodr_file, "r") as f:
            od = parse_opendrive(etree.parse(f).getroot())

        # First pass: create all Road and Lane elements
        for road_elem in od.roads:
            road_id = OpenDriveRoadNetwork._elem_id(road_elem)
            road = OpenDriveRoadNetwork.Road(road_id, road_elem, None)
            self._roads[road_id] = road
            for section_elem in road_elem.lanes.lane_sections:
                for lane_elem in section_elem.leftLanes + section_elem.rightLanes:
                    lane_id = OpenDriveRoadNetwork._elem_id(lane_elem)
                    lane = OpenDriveRoadNetwork.Lane(lane_id, road, lane_elem.id)
                    self._lanes[lane_id] = lane
        
        # Second pass: fill in references
        for road_elem in od.roads:
            road_id = OpenDriveRoadNetwork._elem_id(road_elem)
            road = self._roads[road_id]
            for section_elem in road_elem.lanes.lane_sections:
                for lane_elem in section_elem.leftLanes + section_elem.rightLanes:
                    lane_id = OpenDriveRoadNetwork._elem_id(lane_elem)
                    lane = self._lanes[lane_id]

                    if not lane.in_junction:
                        # When not in an intersection, all OpenDrive Lanes for a Road
                        # with the same index sign go in the same direction.
                        sign = np.sign(lane.index)
                        elems = [elem for elem in section_elem.allLanes if np.sign(elem.id) == sign and elem.id != lane.index]
                        same_dir_lanes = [self._lanes[OpenDriveRoadNetwork._elem_id(elem)] for elem in elems]
                        lane.lanes_in_same_direction = same_dir_lanes
                    else:
                        pass

                        # TODO
                        
                        # result = []
                        # in_roads = set(il.road for il in self.incoming_lanes)
                        # out_roads = set(il.road for il in self.outgoing_lanes)
                        # for lane in self.road.lanes:
                        #     if self == lane:
                        #         continue
                        #     other_in_roads = set(il.road for il in lane.incoming_lanes)
                        #     if in_roads & other_in_roads:
                        #         other_out_roads = set(il.road for il in self.outgoing_lanes)
                        #         if out_roads & other_out_roads:
                        #             result.append(lane)
                        # lane.lanes_in_same_direction = result


    def _precompute_junction_connections(self):
        for road in self._network.roads:
            if road.junction:
                for lane in road.lanes.lane_sections[0].allLanes:
                    lane_id = str(road.id) + "_" + str(0) + "_" + str(lane.id)
                    if lane_id not in self._junction_connections:
                        self._junction_connections[lane_id] = [[], []]
                    if lane.link.predecessorId:
                        road_predecessor = road.link.predecessor
                        road_elem = self._network.getRoad(road_predecessor.element_id)
                        last_ls_index = road_elem.lanes.getLastLaneSectionIdx()
                        pred_lane_id = (
                            str(road_predecessor.element_id)
                            + "_"
                            + str(last_ls_index)
                            + "_"
                            + str(lane.link.predecessorId)
                        )
                        if pred_lane_id not in self._junction_connections:
                            self._junction_connections[pred_lane_id] = [[], [lane_id]]
                        else:
                            self._junction_connections[pred_lane_id][1].append(lane_id)

                        self._junction_connections[lane_id][0].append(pred_lane_id)

                    if lane.link.successorId:
                        road_successor = road.link.successor
                        succ_lane_id = (
                            str(road_successor.element_id)
                            + "_"
                            + str(0)
                            + "_"
                            + str(lane.link.successorId)
                        )
                        if succ_lane_id not in self._junction_connections:
                            self._junction_connections[succ_lane_id] = [[lane_id], []]
                        else:
                            self._junction_connections[succ_lane_id][0].append(lane_id)

                        self._junction_connections[lane_id][1].append(succ_lane_id)

    @property
    def junction_connections(self):
        return self._junction_connections

    def get_junction(self, junction_id):
        return self._network.getJunction(junction_id)

    @property
    def source(self) -> str:
        """This is the .xodr file of the OpenDRIVE map."""
        return self._xodr_file

    class Lane(RoadMap.Lane):
        def __init__(self, lane_id: str, road: RoadMap.Road, index: int):
            self._lane_id = lane_id
            self._road = road
            self._index = index
            self._lanes_in_same_dir = []
            # self._map = road_map
            # self._road = road_map.road_by_id(str(lane_elem.parentRoad.id))
            # assert self._road
            # self._lane_elem = lane_elem
            # self._curr_lane_section = self._lane_elem.lane_section
            # self.type = self._lane_elem.type

        @property
        def lane_id(self) -> str:
            return self._lane_id

        @property
        def road(self) -> RoadMap.Road:
            return self._road

        @property
        def in_junction(self) -> bool:
            return self._road.is_junction

        @property
        def index(self) -> int:
            # TODO: convert to expected convention?
            return self._index

        @property
        def lanes_in_same_direction(self) -> List[RoadMap.Lane]:
            return self._lanes_in_same_dir

        @lanes_in_same_direction.setter
        def lanes_in_same_direction(self, lanes):
            self._lanes_in_same_dir = lanes 

        @cached_property
        def lane_to_left(self) -> Tuple[RoadMap.Lane, bool]:
            if self.index == 0:
                return None, True
            result = None
            if self.index > 0:
                for other in self.lanes_in_same_direction:
                    if self.index - other.index == 1:
                        result = other
                        break
            elif self.index < 0:
                for other in self.lanes_in_same_direction:
                    if self.index - other.index == -1:
                        result = other
                        break
            return result, True

        @cached_property
        def lane_to_right(self) -> Tuple[RoadMap.Lane, bool]:
            if self.index == 0:
                return None, True
            result = None
            if self.index > 0:
                for other in self.lanes_in_same_direction:
                    if self.index - other.index == -1:
                        result = other
                        break
            elif self.index < 0:
                for other in self.lanes_in_same_direction:
                    if self.index - other.index == 1:
                        result = other
                        break
            return result, True

        @cached_property
        def incoming_lanes(self) -> List[RoadMap.Lane]:
            il = []
            if self.lane_id in self._map.junction_connections:
                for pred_lane_id in self._map.junction_connections[self.lane_id][0]:
                    il.append(self._map.lane_by_id(pred_lane_id))

            if self.in_junction:
                return il

            lane_link = self._lane_elem.link
            if lane_link.predecessorId:
                ls_index = self._curr_lane_section.idx
                if ls_index == 0:
                    road_predecessor = self._road.predecessor
                    if road_predecessor and road_predecessor.elementType == "road":
                        pred_road = self._map.road_by_id(
                            str(road_predecessor.element_id)
                        )
                        if len(pred_road.lane_sections) > 1:
                            last_ls_index = len(pred_road.lane_sections) - 1
                        else:
                            last_ls_index = 0
                        pred_lane_id = (
                            str(road_predecessor.element_id)
                            + "_"
                            + str(last_ls_index)
                            + "_"
                            + str(lane_link.predecessorId)
                        )
                        il.append(self._map.lane_by_id(pred_lane_id))

                else:
                    pred_lane_id = (
                        self._road.road_id
                        + "_"
                        + str(ls_index - 1)
                        + "_"
                        + str(lane_link.predecessorId)
                    )
                    il.append(self._map.lane_by_id(pred_lane_id))

            return il

        @cached_property
        def outgoing_lanes(self) -> List[RoadMap.Lane]:
            ol = []
            if self.lane_id in self._map.junction_connections:
                for succ_lane_id in self._map.junction_connections[self.lane_id][1]:
                    ol.append(self._map.lane_by_id(succ_lane_id))

            if self.in_junction:
                return ol

            lane_link = self._lane_elem.link
            if lane_link.successorId:
                ls_index = self._curr_lane_section.idx
                if ls_index == len(self._road.lane_sections) - 1:
                    road_successor = self._road.successor
                    if road_successor:
                        if road_successor.elementType == "road":
                            succ_lane_id = (
                                str(road_successor.element_id)
                                + "_"
                                + str(0)
                                + "_"
                                + str(lane_link.successorId)
                            )
                            ol.append(self._map.lane_by_id(succ_lane_id))
                else:
                    succ_lane_id = (
                        self._road.road_id
                        + "_"
                        + str(ls_index + 1)
                        + "_"
                        + str(lane_link.successorId)
                    )
                    ol.append(self._map.lane_by_id(succ_lane_id))

            return ol

    def lane_by_id(self, lane_id: str) -> RoadMap.Lane:
        lane = self._lanes.get(lane_id)
        if not lane:
            self._log.warning(
                f"OpenDriveRoadNetwork got request for unknown lane_id '{lane_id}'"
            )
        return lane

    class Road(RoadMap.Road):
        def __init__(self, road_id: str, road_elem: RoadElement, road_map):
            self._road_id = road_id
            self._road_elem = road_elem
            self._predecessor_elem = self._road_elem.link.predecessor
            self._successor_elem = self._road_elem.link.successor
            self._lane_sections = self._road_elem.lanes.lane_sections
            self._map = road_map

        def _find_junction_connections(self, junction):
            junction_conns = []
            for connection in junction.connections:
                if connection.incomingRoad == self.road_id:
                    junction_conns.append(
                        self._map.road_by_id(str(connection.connectingRoad))
                    )
            return junction_conns

        @property
        def predecessor(self):
            return self._predecessor_elem

        @property
        def successor(self):
            return self._successor_elem

        @property
        def lane_sections(self):
            return self._lane_sections

        @cached_property
        def left_lanes(self) -> List[RoadMap.Lane]:
            left_lanes = []
            for i in range(len(self._lane_sections)):
                for od_lane in self._lane_sections[i].leftLanes:
                    lane_id = self.road_id + "_" + str(i) + "_" + str(od_lane.id)
                    left_lanes.append(self._map.lane_by_id(lane_id))
            return left_lanes

        @cached_property
        def right_lanes(self) -> List[RoadMap.Lane]:
            right_lanes = []
            for i in range(len(self._lane_sections)):
                for od_lane in self._lane_sections[i].rightLanes:
                    lane_id = self.road_id + "_" + str(i) + "_" + str(od_lane.id)
                    right_lanes.append(self._map.lane_by_id(lane_id))
            return right_lanes

        @cached_property
        def centre_lanes(self) -> List[RoadMap.Lane]:
            centre_lanes = []
            for i in range(len(self._lane_sections)):
                for od_lane in self._lane_sections[i].centreLanes:
                    lane_id = self.road_id + "_" + str(i) + "_" + str(od_lane.id)
                    centre_lanes.append(self._map.lane_by_id(lane_id))
            return centre_lanes

        # RoadMap API methods

        @property
        def road_id(self) -> str:
            return self._road_id

        @cached_property
        def is_junction(self) -> bool:
            if self._road_elem.junction:
                return True
            return False

        @cached_property
        def length(self) -> float:
            return self._road_elem._length

        @cached_property
        def incoming_roads(self) -> List[RoadMap.Road]:
            in_roads = []
            if self._predecessor_elem:
                if self._predecessor_elem.elementType == "road":
                    in_roads.append(
                        self._map.road_by_id(str(self._predecessor_elem.element_id))
                    )
                elif self._predecessor_elem.elementType == "junction":
                    junction = self._map.get_junction(self._predecessor_elem.element_id)
                    in_roads.extend(self._find_junction_connections(junction))

            return in_roads

        @cached_property
        def outgoing_roads(self) -> List[RoadMap.Road]:
            og_roads = []
            if self._successor_elem:
                if self._successor_elem.elementType == "road":
                    og_roads.append(
                        self._map.road_by_id(str(self._successor_elem.element_id))
                    )
                elif self._successor_elem.elementType == "junction":
                    junction = self._map.get_junction(self._successor_elem.element_id)
                    og_roads.extend(self._find_junction_connections(junction))
            return og_roads

        @cached_property
        def lanes(self) -> List[RoadMap.Lane]:
            lanes = []
            for i in range(len(self._lane_sections)):
                for od_lane in self._lane_sections[i].allLanes:
                    lane_id = self.road_id + "_" + str(i) + "_" + str(od_lane.id)
                    lanes.append(self._map.lane_by_id(lane_id))
            return lanes

        def lane_at_index(self, index: int) -> RoadMap.Lane:
            return self.lanes[index]

    def road_by_id(self, road_id: str) -> RoadMap.Road:
        road = self._roads.get(road_id)
        if not road:
            self._log.warning(
                f"OpenDriveRoadNetwork got request for unknown road_id '{road_id}'"
            )
        return road
