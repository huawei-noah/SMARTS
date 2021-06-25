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
import math
import os
import random
import re
from functools import lru_cache
from subprocess import check_output
from tempfile import NamedTemporaryFile
from typing import List, NamedTuple, Sequence, Tuple, Union

import numpy as np
import trimesh
import trimesh.scene
from cached_property import cached_property
from shapely import ops
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.geometry.base import CAP_STYLE, JOIN_STYLE
from shapely.ops import snap, triangulate
from trimesh.exchange import gltf

from .coordinates import BoundingBox, Point, Pose, RefLinePoint
from .road_map import RoadMap
from .sumo_lanepoints import SumoLanePoints
from .utils.math import fast_quaternion_from_angle, vec_to_radians

from smarts.core.utils.sumo import sumolib  # isort:skip
from sumolib.net.edge import Edge  # isort:skip
from sumolib.net.lane import Lane  # isort:skip


def _convert_camera(camera):
    result = {
        "name": camera.name,
        "type": "perspective",
        "perspective": {
            "aspectRatio": camera.fov[0] / camera.fov[1],
            "yfov": np.radians(camera.fov[1]),
            "znear": float(camera.z_near),
            # HACK: The trimesh gltf export doesn't include a zfar which Panda3D GLB
            #       loader expects. Here we override to make loading possible.
            "zfar": float(camera.z_near + 100),
        },
    }
    return result


gltf._convert_camera = _convert_camera


class _GLBData:
    def __init__(self, bytes_):
        self._bytes = bytes_

    def write_glb(self, output_path):
        with open(output_path, "wb") as f:
            f.write(self._bytes)


class SumoRoadNetwork(RoadMap):
    # 3.2 is the default Sumo road network lane width if it's not specified
    # explicitly in Sumo's NetEdit or the map.net.xml file.
    # This corresponds on a 1:1 scale to lanes 3.2m wide, which is typical
    # in North America (although US highway lanes are wider at ~3.7m).
    DEFAULT_LANE_WIDTH = 3.2

    def __init__(
        self, graph, net_file, default_lane_width=None, lanepoint_spacing=None
    ):
        self._log = logging.getLogger(self.__class__.__name__)
        self._graph = graph
        self._net_file = net_file
        self._default_lane_width = (
            default_lane_width
            if default_lane_width is not None
            else SumoRoadNetwork.DEFAULT_LANE_WIDTH
        )
        self._lanes = {}
        self._roads = {}
        self.lanepoint_spacing = lanepoint_spacing
        self.lanepoints = None
        if lanepoint_spacing is not None:
            assert lanepoint_spacing > 0
            # XXX: this should be last here since SumoLanePoints() calls road_network methods immediately
            self.lanepoints = SumoLanePoints(self, spacing=lanepoint_spacing)

    @staticmethod
    def _check_net_origin(bbox):
        assert len(bbox) == 4
        return bbox[0] <= 0.0 and bbox[1] <= 0.0 and bbox[2] >= 0.0 and bbox[3] >= 0.0

    shifted_net_file_name = "shifted_map-AUTOGEN.net.xml"

    @classmethod
    def shifted_net_file_path(cls, net_file_path):
        net_file_folder = os.path.dirname(net_file_path)
        return os.path.join(net_file_folder, cls.shifted_net_file_name)

    @classmethod
    @lru_cache(maxsize=1)
    def _shift_coordinates(cls, net_file_path, shifted_path):
        assert shifted_path != net_file_path
        logger = logging.getLogger(cls.__name__)
        logger.info(f"normalizing net coordinates into {shifted_path}...")
        ## Translate the map's origin to remove huge (imprecise) offsets.
        ## See https://sumo.dlr.de/docs/netconvert.html#usage_description
        ## for netconvert options description.
        try:
            stdout = check_output(
                [
                    "netconvert",
                    "--offset.disable-normalization=FALSE",
                    "-s",
                    net_file_path,
                    "-o",
                    shifted_path,
                ]
            )
            logger.debug(f"netconvert output: {stdout}")
            return True
        except Exception as e:
            logger.warning(
                f"unable to use netconvert tool to normalize coordinates: {e}"
            )
        return False

    @classmethod
    def from_file(
        cls,
        net_file,
        shift_to_origin=False,
        default_lane_width=None,
        lanepoint_spacing=None,
    ):
        # Connections to internal lanes are implicit. If `withInternal=True` is
        # set internal junctions and the connections from internal lanes are
        # loaded into the network graph.
        G = sumolib.net.readNet(net_file, withInternal=True)

        if not cls._check_net_origin(G.getBoundary()):
            shifted_net_file = cls.shifted_net_file_path(net_file)
            if os.path.isfile(shifted_net_file) or (
                shift_to_origin and cls._shift_coordinates(net_file, shifted_net_file)
            ):
                G = sumolib.net.readNet(shifted_net_file, withInternal=True)
                assert cls._check_net_origin(G.getBoundary())
                net_file = shifted_net_file
                # keep track of having shifted the graph by
                # injecting state into the network graph.
                # this is needed because some maps have been pre-shifted,
                # and will already have a locationOffset, but for those
                # the offset should not be used (because all their other
                # coordinates are relative to the origin).
                G._shifted_by_smarts = True

        return cls(
            G,
            net_file,
            default_lane_width=default_lane_width,
            lanepoint_spacing=lanepoint_spacing,
        )

    # TODO:  get rid of this, fix traffic_history_provider
    @cached_property
    def net_offset(self):
        """ This is our offset from what's in the original net file. """
        return (
            self._graph.getLocationOffset()
            if self._graph and getattr(self._graph, "_shifted_by_smarts", False)
            else [0, 0]
        )

    @property
    def source(self) -> str:
        """ This is the net.xml file that corresponds with our possibly-offset coordinates. """
        return self._net_file

    @cached_property
    def bounding_box(self) -> BoundingBox:
        # maps are assumed to start at the origin
        bb = self._graph.getBoundary()  # 2D bbox in format (xmin, ymin, xmax, ymax)
        return BoundingBox(
            min_pt=Point(x=bb[0], y=bb[1]), max_pt=Point(x=bb[2], y=bb[3])
        )

    @property
    def scale_factor(self) -> float:
        # map units per meter
        return self._default_lane_width / DEFAULT_ENTRY_TACTIC

    def to_glb(self, at_path):
        """ build a glb file for camera rendering and envision """
        polys = self._compute_road_polygons()
        glb = self._make_glb_from_polys(polys)
        glb.write_glb(at_path)

    class Lane(RoadMap.Lane):
        def __init__(self, lane_id: str, sumo_lane, road_map):
            self._lane_id = lane_id
            self._sumo_lane = sumo_lane
            self._map = road_map
            self._road = road_map.road_by_id(sumo_lane.getEdge().getID())
            assert self._road
            self._road_dir = None

        @property
        def lane_id(self) -> str:
            return self._lane_id

        @property
        def road(self) -> RoadMap.Road:
            return self._road

        @property
        def road_dir(self) -> bool:
            if self._road_dir is None and self._road:
                self._road_dir = self._road.lanes[self.index].lane_id == lane_id
            return self._road_dir

        @cached_property
        def index(self) -> int:
            """ 0 is outer / right-most (relative to lane heading) lane on road. """
            return self._sumo_lane.getIndex()

        @cached_property
        def lane_to_left(self) -> Tuple[RoadMap.Lane, bool]:
            """Note: left is defined as 90 degrees clockwise relative to the lane heading.
            Second result is True if lane is in the same direction as this one.
            May return None for lanes in junctions."""
            lanes = self._road.lanes_by_direction(self.road_dir)
            index = self.index + 1
            return (lanes[index], True) if index < len(lanes) else (None, True)

        @cached_property
        def lane_to_right(self) -> Tuple[RoadMap.Lane, bool]:
            """Note: right is defined as 90 degrees counter-clockwise relative to the lane heading.
            Second result is True if lane is in the same direction as this one.
            May return None for lanes in junctions."""
            lanes = self._road.lanes_by_direction(self.road_dir)
            index = self.index - 1
            return (lanes[index], True) if index >= 0 else (None, True)

        @cached_property
        def speed_limit(self) -> float:
            return self._sumo_lane.getSpeed()

        @cached_property
        def length(self) -> float:
            return self._sumo_lane.getLength()

        @cached_property
        def _width(self) -> float:
            return self._sumo_lane.getWidth()

        @property
        def in_junction(self) -> bool:
            """ will return None if not in a junction"""
            return self._road.is_junction

        @cached_property
        def incoming_lanes(self) -> List[RoadMap.Lane]:
            return [
                self._map.lane_by_id(incoming.getID())
                for incoming in self._sumo_lane.getIncoming()
            ]

        @cached_property
        def outgoing_lanes(self) -> List[RoadMap.Lane]:
            return [
                self._map.lane_by_id(outgoing.getID())
                for outgoing in self._sumo_lane.getOutgoing()
            ]

        @cached_property
        def oncoming_lanes(self) -> List[RoadMap.Lane]:
            result = []
            for oncr in self.road.oncoming_roads:
                result += oncr.lanes
            return result

        @lru_cache(maxsize=8)
        def point_in_lane(self, point: Point) -> bool:
            lane_point = self.to_lane_coord(point)
            return abs(lane_point.t) <= self._width / 2

        @lru_cache(maxsize=8)
        def center_at_point(self, point: Point) -> Point:
            offset = self.offset_along_lane(point)
            return self.from_lane_coord(RefLinePoint(s=offset))

        def width_at_offset(self, offset: float) -> float:
            return self._width

        @lru_cache(maxsize=8)
        def edges_at_point(self, point: Point) -> Tuple[Point, Point]:
            offset = self.offset_along_lane(point)
            left_edge = RefLanePoint(s=offset, t=self._width / 2)
            right_edge = RefLanePoint(s=offset, t=-self._width / 2)
            return (self.from_lane_coord(left_edge), self.from_lane_coord(right_edge))

        @lru_cache(maxsize=8)
        def vector_at_offset(self, start_offset: float) -> np.ndarray:
            add_offset = 1
            end_offset = start_offset + add_offset  # a little further down the lane
            length = self._sumo_lane.getLength()
            if end_offset > length + add_offset:
                self._map._log.warning(
                    f"Offset={end_offset} goes out further than the end of lane=({self.lane_id}, length={length})"
                )
            p1 = self.from_lane_coord(RefLinePoint(s=start_offset))
            p2 = self.from_lane_coord(RefLinePoint(s=end_offset))
            return np.array(p2) - np.array(p1)

        @lru_cache(maxsize=8)
        def target_pose_at_point(self, point: Point) -> Pose:
            offset = self.offset_along_lane(point)
            position = self.from_lane_coord(RefLinePoint(s=offset))
            desired_vector = self.vector_at_offset(offset)
            orientation = fast_quaternion_from_angle(vec_to_radians(desired_vector[:2]))
            return Pose(position=position, orientation=orientation)

        @lru_cache(maxsize=8)
        def to_lane_coord(self, world_point: Point) -> RefLinePoint:
            s = self.offset_along_lane(point)
            vector = self.vector_at_offset(s)
            normal = np.array([-vector[1], vector[0]])
            center_at_s = self.from_lane_coord(RefLinePoint(s=s))
            offcenter_vector = np.array(world_point) - center_at_s
            t_sign = np.sign(np.dot(offcenter_vector, normal))
            t = np.linalg.norm(offcenter_vector) * t_sign
            return RefLinePoint(s=u, t=t)

        @lru_cache(maxsize=8)
        def from_lane_coord(self, lane_point: RefLinePoint) -> Point:
            shape = self._sumo_lane.getShape(False)
            x, y = sumolib.geomhelper.positionAtShapeOffset(shape, lane_point.s)
            return Point(x=x, y=y)

        @lru_cache(maxsize=8)
        def offset_along_lane(self, world_point: Point) -> float:
            shape = self._sumo_lane.getShape(False)
            point = world_point[:2]
            if point not in shape:
                return sumolib.geomhelper.polygonOffsetWithMinimumDistanceToPoint(
                    point, shape, perpendicular=False
                )
            # SUMO geomhelper.polygonOffset asserts when the point is part of the shape.
            # We get around the assertion with a check if the point is part of the shape.
            offset = 0
            for i in range(len(shape) - 1):
                if shape[i] == point:
                    break
                offset += sumolib.geomhelper.distance(shape[i], shape[i + 1])
            return offset

        def buffered_shape(self, width: float = 1.0) -> Polygon:
            return SumoRoadNetwork._buffered_shape(self._sumo_lane.getShape(), width)

    def lane_by_id(self, lane_id: str) -> RoadMap.Lane:
        lane = self._lanes.get(lane_id)
        if lane:
            return lane
        sumo_lane = self._graph.getLane(lane_id)
        if not sumo_lane:
            return None
        lane = SumoRoadNetwork.Lane(lane_id, sumo_lane, self)
        self._lanes[lane_id] = lane
        return lane

    class Road(RoadMap.Road):
        def __init__(self, road_id: str, sumo_edge: Edge, road_map):
            self._road_id = road_id
            self._sumo_edge = sumo_edge
            self._map = road_map
            self._lanes = []

        @cached_property
        def is_junction(self) -> bool:
            return self._sumo_edge.isSpecial()

        @cached_property
        def length(self) -> int:
            return self._sumo_edge.getLength()

        @property
        def road_id(self) -> str:
            return self._road_id

        @cached_property
        def incoming_roads(self) -> List[RoadMap.Road]:
            return [
                self._map.road_by_id(edge.getID())
                for edge in self._sumo_edge.getIncoming().keys()
            ]

        @cached_property
        def outgoing_roads(self) -> List[RoadMap.Road]:
            return [
                self._map.road_by_id(edge.getID())
                for edge in self._sumo_edge.getOutgoing().keys()
            ]

        @cached_property
        def oncoming_roads(self) -> List[RoadMap.Road]:
            from_node = self._sumo_edge.getFromNode()
            to_node = self._sumo_edge.getToNode()
            return [
                self._map.road_by_id(edge.getID())
                for edge in to_node.getOutgoing()
                if edge.getToNode().getID() == from_node.getID()
            ]

        @cached_property
        def parallel_roads(self) -> List[RoadMap.Road]:
            from_node, to_node = (
                self._sumo_edge.getFromNode(),
                self._sumo_edge.getToNode(),
            )
            return [
                self._map.road_by_id(edge.getID())
                for edge in from_node.getOutgoing()
                if self.road_id != edge.getID()
                and edge.getToNode().getID() == to_node.getID()
            ]

        @property
        def lanes(self) -> List[RoadMap.Lane]:
            # Note:  all SUMO Lanes for an Edge go in the same direction.
            if not self._lanes:
                self._lanes = [
                    self._map.lane_by_id(sumo_lane.getID())
                    for sumo_lane in self._sumo_edge.getLanes()
                ]
            return self._lanes

        def lane_at_index(self, index: int, direction: bool = True) -> RoadMap.Lane:
            # Note:  all SUMO Lanes for an Edge go in the same direction.
            assert direction
            return self.lanes_by_direction(direction)[index]

        def lanes_by_direction(self, direction: bool) -> List[RoadMap.Lane]:
            """Lanes returned in order of lane index (right-to-left) for a direction.
            direction is arbitrary indicator:
            all True lanes go in the same direction, as do all False lanes,
            but True and False lanes go in opposing directions."""
            # Note:  all SUMO Lanes for an Edge go in the same direction.
            if not self._lanes and direction:
                self._lanes = [
                    self._map.lane_by_id(sumo_lane.getID())
                    for sumo_lane in self._sumo_edge.getLanes()
                ]
            return self._lanes

        @lru_cache(maxsize=8)
        def point_on_road(self, point: Point) -> bool:
            for lane in self.lanes:
                if lane.point_in_lane(point):
                    return True
            return False

        @lru_cache(maxsize=8)
        def edges_at_point(self, point: Point) -> Tuple[Point, Point]:
            lanes = self.lanes
            _, right_edge = lanes[0].edges_at_point(point)
            left_edge, _ = lanes[-1].edges_at_point(point)
            return left_edge, right_edge

        def buffered_shape(self, width: float = 1.0) -> Polygon:
            return SumoRoadNetwork._buffered_shape(self._sumo_edge.getShape(), width)

    def road_by_id(self, road_id: str) -> RoadMap.Road:
        road = self._roads.get(road_id)
        if road:
            return road
        sumo_edge = self._graph.getEdge(road_id)
        if not sumo_edge:
            return None
        road = SumoRoadNetwork.Road(road_id, sumo_edge, self)
        self._roads[road_id] = road
        return road

    @lru_cache(maxsize=16)
    def nearest_lanes(
        self, point: Point, radius: float = None, include_junctions=True
    ) -> List[Tuple[RoadMap.Lane, float]]:
        if radius is None:
            radius = max(10, 2 * self._default_lane_width)
        # XXX: note that this getNeighboringLanes() call is fairly heavy/expensive (as revealed by profiling)
        # XXX: Not robust around junctions (since their shape is quite crude?)
        candidate_lanes = self._graph.getNeighboringLanes(
            point[0],
            point[1],
            r=radius,
            includeJunctions=include_junctions,
            allowFallback=False,
        )
        if not include_junctions:
            candidate_lanes = [
                lane for lane in candidate_lanes if not lane[0].getEdge().isSpecial()
            ]
        candidate_lanes.sort(key=lambda lane_dist_tup: lane_dist_tup[1])
        return [(self.lane_by_id(lane.getID()), dist) for lane, dist in candidate_lanes]

    @lru_cache(maxsize=16)
    def road_with_point(self, point: Point) -> RoadMap.Road:
        radius = max(5, 2 * self._default_lane_width)
        for nl, dist in self.nearest_lanes(point, radius):
            if dist < 0.5 * nl._width + 1e-1:
                return nl.road
        return None

    class Route(RoadMap.Route):
        def __init__(self):
            self._roads = []
            self._length = 0

        @property
        def roads(self) -> List[RoadMap.Road]:
            return self._roads

        @property
        def road_length(self) -> float:
            return self._length

        def add_road(self, road: RoadMap.Road):
            self._length += road.length
            self._roads.append(road)

        @cached_property
        def geometry(self) -> Sequence[Sequence[Tuple[float, float]]]:
            return [
                road.buffered_shape(sum([lane._width for lane in road.lanes]))
                for road in self.roads
            ]

    def generate_routes(
        self,
        start_road: RoadMap.Road,
        end_road: RoadMap.Road,
        via: Sequence[RoadMap.Road] = None,
        max_to_gen: int = None,
    ) -> List[RoadMap.Route]:
        assert max_to_gen == 1, "multiple route generation not yet supported for Sumo"
        newroute = SumoRoadNetwork.Route()
        result = [newroute]

        roads = [start_road]
        if via:
            roads += via
        if end_road != start_road:
            roads.append(end_road)

        edges = []
        for cur_road, next_road in zip(roads, roads[1:] + [None]):
            if not next_road:
                edges.append(cur_road._sumo_edge)
                break
            sub_route = (
                self._graph.getShortestPath(cur_road._sumo_edge, next_road._sumo_edge)[
                    0
                ]
                or []
            )
            assert (
                len(sub_route) >= 2
            ), f"Unable to find valid path (len={len(sub_route)}) between {(cur_road.road_id, next_road.road_id)}"
            # The sub route includes the boundary roads (cur_road, next_road).
            # We clip the latter to prevent duplicates
            edges.extend(sub_route[:-1])

        if len(edges) == 1:
            # route is within a single road
            newroute.add_road(self.road_by_id(edges[0].getID()))
            return result

        used_edges = []
        edge_ids = []
        adjacent_edge_pairs = zip(edges, edges[1:])
        for cur_edge, next_edge in adjacent_edge_pairs:
            internal_routes = self._internal_routes_between(cur_edge, next_edge)
            for internal_route in internal_routes:
                used_edges.extend(internal_route)
                edge_ids.extend([edge.getID() for edge in internal_route])
        _, indices = np.unique(edge_ids, return_index=True)
        for idx in sorted(indices):
            newroute.add_road(self.road_by_id(used_edges[idx].getID()))

        return result

    def _internal_routes_between(
        self, start_edge: Edge, end_edge: Edge
    ) -> List[List[Edge]]:
        routes = []
        outgoing = start_edge.getOutgoing()
        assert end_edge in outgoing, (
            f"{end_edge.getID()} not in {[e.getID() for e in outgoing.keys()]}. "
            "Perhaps you're using a LapMission on a route that is not a closed loop?"
        )
        connections = outgoing[end_edge]
        for connection in connections:
            conn_route = [start_edge]
            # This connection may have some intermediate 'via' lanes.
            # we need to follow these to eventually leave the junction.
            via_lane_id = connection.getViaLaneID()
            while via_lane_id:
                via_lane = self.lane_by_id(via_lane_id)
                via_road = via_lane.road
                via_edge = via_road._sumo_edge

                conn_route.append(via_edge)

                # Sometimes we get the same via lane id multiple times.
                # We convert to a set to remove duplicates.
                next_via_lane_ids = set(
                    conn.getViaLaneID() for conn in via_edge.getOutgoing()[end_edge]
                )
                assert (
                    len(next_via_lane_ids) == 1
                ), f"Expected exactly one next via lane id at {via_lane_id}, got: {next_via_lane_ids}"
                via_lane_id = list(next_via_lane_ids)[0]

            conn_route.append(end_edge)
            routes.append(conn_route)
        return routes

    def random_route(self, max_route_len: int = 10) -> RoadMap.Route:
        route = SumoRoadNetwork.Route()
        next_edges = self._graph.getEdges(False)
        while next_edges and len(route.roads) < max_route_len:
            cur_edge = random.choice(next_edges)
            route.add_road(self.road_by_id(cur_edge.getID()))
            next_edges = list(cur_edge.getOutgoing().keys())
        return route

    def _compute_road_polygons(self):
        lane_to_poly = {}
        for edge in self._graph.getEdges():
            for lane in edge.getLanes():
                shape = SumoRoadNetwork._buffered_shape(
                    lane.getShape(), lane.getWidth()
                )
                # Check if "shape" is just a point.
                if len(set(shape.exterior.coords)) == 1:
                    logging.debug(
                        f"Lane:{lane.getID()} has provided non-shape values {lane.getShape()}"
                    )
                    continue

                lane_to_poly[lane.getID()] = shape

        # Remove holes created at tight junctions due to crude map geometry
        self._snap_internal_holes(lane_to_poly)
        self._snap_external_holes(lane_to_poly)
        # Remove break in visible lane connections created when lane enters an intersection
        self._snap_internal_edges(lane_to_poly)

        polys = list(lane_to_poly.values())

        for node in self._graph.getNodes():
            line = node.getShape()
            if len(line) <= 2 or len(set(line)) == 1:
                self._log.debug(
                    "Skipping {}-type node with <= 2 vertices".format(node.getType())
                )
                continue

            polys.append(Polygon(line))

        return polys

    def _snap_internal_edges(self, lane_to_poly, snap_threshold=2):
        # HACK: Internal edges that have tight curves, when buffered their ends do not
        #       create a tight seam with the connected lanes. This procedure attempts
        #       to remedy that with snapping.
        for lane_id in lane_to_poly:
            lane = self._graph.getLane(lane_id)

            # Only do snapping for internal edge lanes
            if not lane.getEdge().isSpecial():
                continue

            lane_shape = lane_to_poly[lane_id]
            incoming = self._graph.getLane(lane_id).getIncoming()[0]
            incoming_shape = lane_to_poly.get(incoming.getID())
            if incoming_shape:
                lane_shape = Polygon(snap(lane_shape, incoming_shape, snap_threshold))
                lane_to_poly[lane_id] = lane_shape

            outgoing = self._graph.getLane(lane_id).getOutgoing()[0].getToLane()
            outgoing_shape = lane_to_poly.get(outgoing.getID())
            if outgoing_shape:
                lane_shape = Polygon(snap(lane_shape, outgoing_shape, snap_threshold))
                lane_to_poly[lane_id] = lane_shape

    def _snap_internal_holes(self, lane_to_poly, snap_threshold=2):
        for lane_id in lane_to_poly:
            lane = self._graph.getLane(lane_id)

            # Only do snapping for internal edge lane holes
            if not lane.getEdge().isSpecial():
                continue
            lane_shape = lane_to_poly[lane_id]
            for x, y in lane_shape.exterior.coords:
                for nl, dist in self.nearest_lanes(
                    Point(x, y),
                    include_junctions=False,
                ):
                    if not nl:
                        continue
                    nl_shape = lane_to_poly.get(nl.lane_id)
                    if nl_shape:
                        lane_shape = Polygon(snap(lane_shape, nl_shape, snap_threshold))
            lane_to_poly[lane_id] = lane_shape

    def _snap_external_holes(self, lane_to_poly, snap_threshold=2):
        for lane_id in lane_to_poly:
            lane = self._graph.getLane(lane_id)

            # Only do snapping for external edge lane holes
            if lane.getEdge().isSpecial():
                continue

            incoming = self._graph.getLane(lane_id).getIncoming()
            if incoming and incoming[0].getEdge().isSpecial():
                continue

            outgoing = self._graph.getLane(lane_id).getOutgoing()
            if outgoing:
                outgoing_lane = outgoing[0].getToLane()
                if outgoing_lane.getEdge().isSpecial():
                    continue

            lane_shape = lane_to_poly[lane_id]
            for x, y in lane_shape.exterior.coords:
                for nl, dist in self.nearest_lanes(
                    Point(x, y),
                    include_junctions=False,
                ):
                    if (not nl) or (nl and nl.in_junction):
                        continue
                    nl_shape = lane_to_poly.get(nl.lane_id)
                    if nl_shape:
                        lane_shape = Polygon(snap(lane_shape, nl_shape, snap_threshold))
            lane_to_poly[lane_id] = lane_shape

    @staticmethod
    def _triangulate(polygon):
        return [
            tri_face
            for tri_face in triangulate(polygon)
            if tri_face.centroid.within(polygon)
        ]

    def _make_glb_from_polys(self, polygons):
        scene = trimesh.Scene()
        vertices, faces = [], []
        point_dict = dict()
        current_point_index = 0

        # Trimesh's API require a list of vertices and a list of faces, where each
        # face contains three indexes into the vertices list. Ideally, the vertices
        # are all unique and the faces list references the same indexes as needed.
        # TODO: Batch the polygon processing.
        for poly in polygons:
            # Collect all the points on the shape to reduce checks by 3 times
            for x, y in poly.exterior.coords:
                p = (x, y, 0)
                if p not in point_dict:
                    vertices.append(p)
                    point_dict[p] = current_point_index
                    current_point_index += 1
            triangles = SumoRoadNetwork._triangulate(poly)
            for triangle in triangles:
                face = np.array(
                    [point_dict.get((x, y, 0), -1) for x, y in triangle.exterior.coords]
                )
                # Add face if not invalid
                if -1 not in face:
                    faces.append(face)

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        # Trimesh doesn't support a coordinate-system="z-up" configuration, so we
        # have to apply the transformation manually.
        mesh.apply_transform(
            trimesh.transformations.rotation_matrix(math.pi / 2, [-1, 0, 0])
        )

        # Attach additional information for rendering as metadata in the map glb
        metadata = {}

        # <2D-BOUNDING_BOX>: four floats separated by ',' (<FLOAT>,<FLOAT>,<FLOAT>,<FLOAT>),
        # which describe x-minimum, y-minimum, x-maximum, and y-maximum
        metadata["bounding_box"] = self._graph.getBoundary()

        # lane markings information
        lane_dividers, edge_dividers = self._compute_traffic_dividers()
        metadata["lane_dividers"] = lane_dividers
        metadata["edge_dividers"] = edge_dividers

        mesh.visual = trimesh.visual.TextureVisuals(
            material=trimesh.visual.material.PBRMaterial()
        )

        scene.add_geometry(mesh)
        return _GLBData(gltf.export_glb(scene, extras=metadata, include_normals=True))

    @staticmethod
    def _buffered_shape(shape, width: float = 1.0) -> Polygon:
        ls = LineString(shape).buffer(
            width / 2,
            1,
            cap_style=CAP_STYLE.flat,
            join_style=JOIN_STYLE.round,
            mitre_limit=5.0,
        )
        if isinstance(ls, MultiPolygon):
            # Sometimes it oddly outputs a MultiPolygon and then we need to turn it into a convex hull
            ls = ls.convex_hull
        elif not isinstance(ls, Polygon):
            raise RuntimeError("Shapely `object.buffer` behavior may have changed.")
        return ls

    def _compute_traffic_dividers(self, threshold=1):
        lane_dividers = []  # divider between lanes with same traffic direction
        edge_dividers = []  # divider between lanes with opposite traffic direction
        edge_borders = []
        for edge in self._graph.getEdges():
            # Omit intersection for now
            if edge.getFunction() == "internal":
                continue

            lanes = edge.getLanes()
            for i in range(len(lanes)):
                shape = lanes[i].getShape()
                left_side = sumolib.geomhelper.move2side(
                    shape, -lanes[i].getWidth() / 2
                )
                right_side = sumolib.geomhelper.move2side(
                    shape, lanes[i].getWidth() / 2
                )

                if i == 0:
                    edge_borders.append(right_side)

                if i == len(lanes) - 1:
                    edge_borders.append(left_side)
                else:
                    lane_dividers.append(left_side)

        # The edge borders that overlapped in positions form an edge divider
        for i in range(len(edge_borders) - 1):
            for j in range(i + 1, len(edge_borders)):
                edge_border_i = np.array(
                    [edge_borders[i][0], edge_borders[i][-1]]
                )  # start and end position
                edge_border_j = np.array(
                    [edge_borders[j][-1], edge_borders[j][0]]
                )  # start and end position with reverse traffic direction

                # The edge borders of two lanes do not always overlap perfectly, thus relax the tolerance threshold to 1
                if np.linalg.norm(edge_border_i - edge_border_j) < threshold:
                    edge_dividers.append(edge_borders[i])

        return lane_dividers, edge_dividers

    # specific to SUMO road networks
    def get_edge_in_junction(
        self, start_edge_id, start_lane_index, end_edge_id, end_lane_index
    ) -> str:
        start_edge = self._graph.getEdge(start_edge_id)
        start_lane = start_edge.getLane(start_lane_index)
        end_edge = self._graph.getEdge(end_edge_id)
        end_lane = end_edge.getLane(end_lane_index)
        connection = start_lane.getConnection(end_lane)

        # If there is no connection beween try and do the best
        if connection is None:
            # The first id is good enough since we just need to determine the junction edge id
            connection = start_edge.getConnections(end_edge)[0]

        connection_lane_id = connection.getViaLaneID()
        connection_lane = self._graph.getLane(connection_lane_id)

        return connection_lane.getEdge().getID()
