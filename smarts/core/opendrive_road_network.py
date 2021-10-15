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
import time
from typing import Dict, List, Tuple

import numpy as np
from cached_property import cached_property
from lxml import etree
from numpy.core.defchararray import index
from opendrive2lanelet.opendriveparser.elements.opendrive import (
    OpenDrive as OpenDriveElement,
)
from opendrive2lanelet.opendriveparser.elements.road import Road as RoadElement
from opendrive2lanelet.opendriveparser.elements.roadLanes import Lane as LaneElement
from opendrive2lanelet.opendriveparser.elements.junction import (
    Junction as JunctionElement,
)
from opendrive2lanelet.opendriveparser.elements.junction import (
    Connection as ConnectionElement,
)
from opendrive2lanelet.opendriveparser.parser import parse_opendrive

from smarts.core.road_map import RoadMap


class OpenDriveRoadNetwork(RoadMap):
    def __init__(self, xodr_file: str):
        self._log = logging.getLogger(self.__class__.__name__)
        self._log.setLevel(logging.INFO)
        self._xodr_file = xodr_file
        self._roads: Dict[str, OpenDriveRoadNetwork.Road] = {}
        self._lanes: Dict[str, OpenDriveRoadNetwork.Lane] = {}
        self._lanepoints = None
        self._junction_connections = {}

    @classmethod
    def from_file(
        cls,
        xodr_file,
    ):
        od_map = cls(xodr_file)
        od_map.load()
        return od_map

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
        start = time.time()
        od: OpenDriveElement = None
        with open(self._xodr_file, "r") as f:
            od = parse_opendrive(etree.parse(f).getroot())
        end = time.time()
        elapsed = round((end - start) * 1000.0, 3)
        self._log.info(f"Parsing .xodr file: {elapsed} ms")

        # First pass: create all Road and Lane elements, and cache junctions
        start = time.time()
        for road_elem in od.roads:
            road_elem: RoadElement = road_elem
            road_id = OpenDriveRoadNetwork._elem_id(road_elem)
            road = OpenDriveRoadNetwork.Road(
                road_id, road_elem.junction is not None, road_elem.length
            )
            self._roads[road_id] = road
            for section_elem in road_elem.lanes.lane_sections:
                for lane_elem in section_elem.leftLanes + section_elem.rightLanes:
                    lane_id = OpenDriveRoadNetwork._elem_id(lane_elem)
                    lane = OpenDriveRoadNetwork.Lane(
                        lane_id, road, lane_elem.id, section_elem.length
                    )
                    self._lanes[lane_id] = lane

        junction_elems: Dict[str, JunctionElement] = {}
        junction_elem: JunctionElement
        for junction_elem in od.junctions:
            junction_elems[junction_elem.id] = junction_elem

        end = time.time()
        elapsed = round((end - start) * 1000.0, 3)
        self._log.info(f"First pass: {elapsed} ms")

        # Second pass: compute road and lane connections
        start = time.time()
        self._precompute_junction_connections(od)
        for road_elem in od.roads:
            road_id = OpenDriveRoadNetwork._elem_id(road_elem)
            road = self._roads[road_id]

            # Incoming roads - simple case
            if (
                road_elem.link.predecessor
                and road_elem.link.predecessor.elementType == "road"
            ):
                in_road = self.road_by_id(str(road_elem.link.predecessor.element_id))
                road.incoming_roads = [in_road]

            # Outgoing roads - simple case
            if (
                road_elem.link.successor
                and road_elem.link.successor.elementType == "road"
            ):
                out_road = self.road_by_id(str(road_elem.link.successor.element_id))
                road.outgoing_roads = [out_road]

            # Incoming/outgoing roads - junction case
            if (
                road_elem.link.predecessor
                and road_elem.link.predecessor.elementType == "junction"
            ) or (
                road_elem.link.successor
                and road_elem.link.successor.elementType == "junction"
            ):
                in_roads, out_roads = [], []
                junction_elem = junction_elems[road_elem.link.predecessor.element_id]

                # Loop over all roads in junction, and check if they're incoming or outgoing for the current road
                for connection in junction_elem.connections:
                    cr_elem = od.getRoad(connection.connectingRoad)
                    if (
                        cr_elem.link.successor
                        and cr_elem.link.successor.element_id == int(road_id)
                    ):
                        in_roads.append(self.road_by_id(str(connection.connectingRoad)))
                    if (
                        cr_elem.link.predecessor
                        and cr_elem.link.predecessor.element_id == int(road_id)
                    ):
                        out_roads.append(
                            self.road_by_id(str(connection.connectingRoad))
                        )
                road.incoming_roads = in_roads
                road.outgoing_roads = out_roads

            for section_elem in road_elem.lanes.lane_sections:
                for lane_elem in section_elem.leftLanes + section_elem.rightLanes:
                    lane_id = OpenDriveRoadNetwork._elem_id(lane_elem)
                    lane = self._lanes[lane_id]
                    road.lanes.append(lane)

                    # Compute incoming lanes
                    lane.incoming_lanes = self._compute_incoming_lane_connections(
                        od, lane, lane_elem, road_elem
                    )

                    # Compute outgoing lanes
                    lane.outgoing_lanes = self._compute_outgoing_lane_connections(
                        lane, lane_elem, road_elem
                    )
        end = time.time()
        elapsed = round((end - start) * 1000.0, 3)
        self._log.info(f"Second pass: {elapsed} ms")

        # Third pass: fill in remaining properties
        start = time.time()
        for road_elem in od.roads:
            road_id = OpenDriveRoadNetwork._elem_id(road_elem)
            road = self._roads[road_id]

            # TODO: compute road dependent attributes

            for section_elem in road_elem.lanes.lane_sections:
                for lane_elem in section_elem.leftLanes + section_elem.rightLanes:
                    lane_id = OpenDriveRoadNetwork._elem_id(lane_elem)
                    lane = self._lanes[lane_id]

                    # Compute lanes in same direction
                    if not lane.in_junction:
                        # When not in an intersection, all OpenDrive Lanes for a Road
                        # with the same index sign go in the same direction.
                        sign = np.sign(lane.index)
                        elems = [
                            elem
                            for elem in section_elem.allLanes
                            if np.sign(elem.id) == sign and elem.id != lane.index
                        ]
                        same_dir_lanes = [
                            self._lanes[OpenDriveRoadNetwork._elem_id(elem)]
                            for elem in elems
                        ]
                        lane.lanes_in_same_direction = same_dir_lanes
                    else:
                        result = []
                        in_roads = set(il.road for il in lane.incoming_lanes)
                        out_roads = set(il.road for il in lane.outgoing_lanes)
                        for rd_lane in lane.road.lanes:
                            if self == rd_lane:
                                continue
                            other_in_roads = set(
                                il.road for il in rd_lane.incoming_lanes
                            )
                            if in_roads & other_in_roads:
                                other_out_roads = set(
                                    il.road for il in lane.outgoing_lanes
                                )
                                if out_roads & other_out_roads:
                                    result.append(rd_lane)
                        lane.lanes_in_same_direction = result

                    # Compute lane to the left
                    result = None
                    if lane.index > 0:
                        for other in lane.lanes_in_same_direction:
                            if lane.index - other.index == 1:
                                result = other
                                break
                    elif lane.index < 0:
                        for other in lane.lanes_in_same_direction:
                            if lane.index - other.index == -1:
                                result = other
                                break
                    lane.lane_to_left = result, True

                    # Compute lane to right
                    result = None
                    if lane.index > 0:
                        for other in lane.lanes_in_same_direction:
                            if lane.index - other.index == -1:
                                result = other
                                break
                    elif lane.index < 0:
                        for other in lane.lanes_in_same_direction:
                            if lane.index - other.index == 1:
                                result = other
                                break
                    lane.lane_to_right = result, True

                    # Compute lane foes
                    result = [
                        incoming
                        for outgoing in lane.outgoing_lanes
                        for incoming in outgoing.incoming_lanes
                        if incoming != lane
                    ]
                    if lane.in_junction:
                        in_roads = set(il.road for il in lane.incoming_lanes)
                        for foe in lane.road.lanes:
                            foe_in_roads = set(il.road for il in foe.incoming_lanes)
                            if not bool(in_roads & foe_in_roads):
                                result.append(foe)
                    lane.foes = list(set(result))

        end = time.time()
        elapsed = round((end - start) * 1000.0, 3)
        self._log.info(f"Third pass: {elapsed} ms")

    def _compute_incoming_lane_connections(
        self, od, lane, lane_elem, road_elem
    ) -> List[RoadMap.Lane]:
        il = []
        if lane.lane_id in self._junction_connections:
            for pred_lane_id in self._junction_connections[lane.lane_id][0]:
                il.append(self.lane_by_id(pred_lane_id))

        if lane.in_junction:
            return il

        lane_link = lane_elem.link
        if lane_link.predecessorId:
            ls_index = lane_elem.lane_section.idx
            if ls_index == 0:
                road_predecessor = road_elem.link.predecessor
                if road_predecessor and road_predecessor.elementType == "road":
                    pred_road_elem = od.getRoad(road_predecessor.element_id)
                    last_ls_index = pred_road_elem.lanes.getLastLaneSectionIdx()
                    pred_lane_id = f"{road_predecessor.element_id}_{last_ls_index}_{lane_link.predecessorId}"
                    il.append(self.lane_by_id(pred_lane_id))

            else:
                pred_lane_id = (
                    f"{road_elem.id}_{ls_index - 1}_{lane_link.predecessorId}"
                )
                il.append(self.lane_by_id(pred_lane_id))

        return il

    def _compute_outgoing_lane_connections(
        self, lane, lane_elem, road_elem
    ) -> List[RoadMap.Lane]:
        ol = []
        if lane.lane_id in self._junction_connections:
            for succ_lane_id in self._junction_connections[lane.lane_id][1]:
                ol.append(self.lane_by_id(succ_lane_id))

        if lane.in_junction:
            return ol

        lane_link = lane_elem.link
        if lane_link.successorId:
            ls_index = lane_elem.lane_section.idx
            if ls_index == len(road_elem.lanes.lane_sections) - 1:
                road_successor = road_elem.link.successor
                if road_successor:
                    if road_successor.elementType == "road":
                        succ_lane_id = (
                            f"{road_successor.element_id}_{0}_{lane_link.successorId}"
                        )
                        ol.append(self.lane_by_id(succ_lane_id))
            else:
                succ_lane_id = f"{road_elem.id}_{ls_index + 1}_{lane_link.successorId}"
                ol.append(self.lane_by_id(succ_lane_id))

        return ol

    def _precompute_junction_connections(self, od: OpenDriveElement):
        for road_elem in od.roads:
            if road_elem.junction:
                for lane_elem in road_elem.lanes.lane_sections[0].allLanes:
                    lane_id = OpenDriveRoadNetwork._elem_id(lane_elem)

                    if lane_id not in self._junction_connections:
                        self._junction_connections[lane_id] = [[], []]

                    if lane_elem.link.predecessorId:
                        road_predecessor = road_elem.link.predecessor
                        pred_road_elem = od.getRoad(road_predecessor.element_id)
                        last_ls_index = pred_road_elem.lanes.getLastLaneSectionIdx()
                        pred_lane_id = f"{road_predecessor.element_id}_{last_ls_index}_{lane_elem.link.predecessorId}"

                        if pred_lane_id not in self._junction_connections:
                            self._junction_connections[pred_lane_id] = [[], [lane_id]]
                        else:
                            self._junction_connections[pred_lane_id][1].append(lane_id)

                        self._junction_connections[lane_id][0].append(pred_lane_id)

                    if lane_elem.link.successorId:
                        road_successor = road_elem.link.successor
                        succ_lane_id = f"{road_successor.element_id}_{0}_{lane_elem.link.successorId}"

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
        def __init__(self, lane_id: str, road: RoadMap.Road, index: int, length: float):
            self._lane_id = lane_id
            self._road = road
            self._index = index
            self._length = length
            self._incoming_lanes = []
            self._outgoing_lanes = []
            self._lanes_in_same_dir = []
            self._foes = []
            self._lane_to_left = None, True
            self._lane_to_right = None, True
            self._in_junction = None

        @property
        def lane_id(self) -> str:
            return self._lane_id

        @property
        def road(self) -> RoadMap.Road:
            return self._road

        @property
        def length(self) -> float:
            return self._length

        @property
        def in_junction(self) -> bool:
            return self.road.is_junction

        @property
        def index(self) -> int:
            # TODO: convert to expected convention?
            return self._index

        @property
        def incoming_lanes(self) -> List[RoadMap.Lane]:
            return self._incoming_lanes

        @incoming_lanes.setter
        def incoming_lanes(self, value):
            self._incoming_lanes = value

        @property
        def outgoing_lanes(self) -> List[RoadMap.Lane]:
            return self._outgoing_lanes

        @outgoing_lanes.setter
        def outgoing_lanes(self, value):
            self._outgoing_lanes = value

        @property
        def lanes_in_same_direction(self) -> List[RoadMap.Lane]:
            return self._lanes_in_same_dir

        @lanes_in_same_direction.setter
        def lanes_in_same_direction(self, lanes):
            self._lanes_in_same_dir = lanes

        @property
        def lane_to_left(self) -> Tuple[RoadMap.Lane, bool]:
            return self._lane_to_left

        @lane_to_left.setter
        def lane_to_left(self, value):
            self._lane_to_left = value

        @property
        def lane_to_right(self) -> Tuple[RoadMap.Lane, bool]:
            return self._lane_to_right

        @lane_to_right.setter
        def lane_to_right(self, value):
            self._lane_to_right = value

        @property
        def foes(self) -> List[RoadMap.Lane]:
            return self._foes

        @foes.setter
        def foes(self, value):
            self._foes = value

    def lane_by_id(self, lane_id: str) -> RoadMap.Lane:
        lane = self._lanes.get(lane_id)
        if not lane:
            self._log.warning(
                f"OpenDriveRoadNetwork got request for unknown lane_id '{lane_id}'"
            )
        return lane

    class Road(RoadMap.Road):
        def __init__(self, road_id: str, is_junction: bool, length: float):
            self._log = logging.getLogger(self.__class__.__name__)
            self._road_id = road_id
            self._is_junction = is_junction
            self._length = length
            self._lanes = []
            self._incoming_roads = []
            self._outgoing_roads = []

        @property
        def road_id(self) -> str:
            return self._road_id

        @property
        def is_junction(self) -> bool:
            return self._is_junction

        @property
        def length(self) -> float:
            return self._length

        @property
        def incoming_roads(self) -> List[RoadMap.Road]:
            return self._incoming_roads

        @incoming_roads.setter
        def incoming_roads(self, value):
            self._incoming_roads = value

        @property
        def outgoing_roads(self) -> List[RoadMap.Road]:
            return self._outgoing_roads

        @outgoing_roads.setter
        def outgoing_roads(self, value):
            self._outgoing_roads = value

        @property
        def parallel_roads(self) -> List[RoadMap.Road]:
            return []

        @property
        def lanes(self) -> List[RoadMap.Lane]:
            return self._lanes

        @lanes.setter
        def lanes(self, value):
            self._lanes = value

        def lane_at_index(self, index: int) -> RoadMap.Lane:
            lanes_with_index = [lane for lane in self.lanes if lane.index == index]
            if len(lanes_with_index) == 0:
                self._log.warning(
                    f"Road with id {self.road_id} has no lane at index {index}"
                )
                return None
            assert len(lanes_with_index) == 1
            return lanes_with_index[0]

    def road_by_id(self, road_id: str) -> RoadMap.Road:
        road = self._roads.get(road_id)
        if not road:
            self._log.warning(
                f"OpenDriveRoadNetwork got request for unknown road_id '{road_id}'"
            )
        return road
