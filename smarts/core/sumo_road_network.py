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
from shapely.geometry import Polygon
from shapely.ops import snap, triangulate
from trimesh.exchange import gltf

from .coordinates import BoundingBox, Heading, Point, Pose, RefLinePoint
from .road_map import RoadMap, Waypoint
from .sumo_lanepoints import LinkedLanePoint, SumoLanePoints
from .utils.geometry import buffered_shape
from .utils.math import (
    inplace_unwrap,
    radians_to_vec,
    vec_2d,
)

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
        self._waypoints_cache = SumoRoadNetwork._WaypointsCache()
        self._lanepoints = None
        if lanepoint_spacing is not None:
            assert lanepoint_spacing > 0
            # XXX: this should be last here since SumoLanePoints() calls road_network methods immediately
            self._lanepoints = SumoLanePoints(self, spacing=lanepoint_spacing)

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
    def xy_offset(self):
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
        return self._default_lane_width / SumoRoadNetwork.DEFAULT_LANE_WIDTH

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

        @property
        def lane_id(self) -> str:
            return self._lane_id

        @property
        def road(self) -> RoadMap.Road:
            return self._road

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
            return self._road.is_junction

        @cached_property
        def index(self) -> int:
            return self._sumo_lane.getIndex()

        @cached_property
        def lanes_in_same_direction(self) -> List[RoadMap.Lane]:
            if not self.in_junction:
                # When not in an intersection, all SUMO Lanes for an Edge go in the same direction.
                return [l for l in self.road.lanes if l != self]
            result = []
            in_roads = set(il.road for il in self.incoming_lanes)
            out_roads = set(il.road for il in self.outgoing_lanes)
            for lane in self.road.lanes:
                if self == lane:
                    continue
                other_in_roads = set(il.road for il in lane.incoming_lanes)
                if in_roads & other_in_roads:
                    other_out_roads = set(il.road for il in self.outgoing_lanes)
                    if out_roads & other_out_roads:
                        result.append(lane)
            return result

        @cached_property
        def lane_to_left(self) -> Tuple[RoadMap.Lane, bool]:
            result = None
            for other in self.lanes_in_same_direction:
                if other.index > self.index and (
                    not result or other.index < result.index
                ):
                    result = other
            return result, True

        @cached_property
        def lane_to_right(self) -> Tuple[RoadMap.Lane, bool]:
            result = None
            for other in self.lanes_in_same_direction:
                if other.index < self.index and (
                    not result or other.index > result.index
                ):
                    result = other
            return result, True

        @cached_property
        def incoming_lanes(self) -> List[RoadMap.Lane]:
            return [
                self._map.lane_by_id(incoming.getID())
                for incoming in self._sumo_lane.getIncoming()
            ]

        @cached_property
        def outgoing_lanes(self) -> List[RoadMap.Lane]:
            return [
                self._map.lane_by_id(outgoing.getToLane().getID())
                for outgoing in self._sumo_lane.getOutgoing()
            ]

        @lru_cache(maxsize=16)
        def oncoming_lanes_at_offset(self, offset: float) -> List[RoadMap.Lane]:
            result = []
            radius = 1.1 * self.width_at_offset(offset)
            pt = self.from_lane_coord(RefLinePoint(offset))
            nearby_lanes = self._map.nearest_lanes(pt, radius=radius)
            if not nearby_lanes:
                return result
            my_vect = self.vector_at_offset(offset)
            my_norm = np.linalg.norm(my_vect)
            threshold = -0.995562  # cos(175*pi/180)
            for lane, _ in nearby_lanes:
                lv = lane.vector_at_offset(offset)
                lane_angle = np.dot(my_vect, lv) / (my_norm * np.linalg.norm(lv))
                if lane_angle < threshold:
                    result.append(lane)
            return result

        @cached_property
        def foes(self) -> List[RoadMap.Lane]:
            # TODO:  we might do better here since Sumo/Traci determines right-of-way for their connections/links.  See:
            #        https://sumo.dlr.de/pydoc/traci._lane.html#LaneDomain-getFoes
            result = [
                incoming
                for outgoing in self.outgoing_lanes
                for incoming in outgoing.incoming_lanes
                if incoming != self
            ]
            if self.in_junction:
                in_roads = set(il.road for il in self.incoming_lanes)
                for foe in self.road.lanes:
                    foe_in_roads = set(il.road for il in foe.incoming_lanes)
                    if not bool(in_roads & foe_in_roads):
                        result.append(foe)
            return list(set(result))

        def waypoint_paths_for_pose(
            self, pose: Pose, lookahead: int, route: RoadMap.Route = None
        ) -> List[List[Waypoint]]:
            road_ids = [road.road_id for road in route.roads] if route else None
            return self._waypoint_paths_at(pose.position, lookahead, road_ids)

        def waypoint_paths_at_offset(
            self, offset: float, lookahead: int = 30, route: RoadMap.Route = None
        ) -> List[List[Waypoint]]:
            wp_start = self.from_lane_coord(RefLinePoint(offset))
            road_ids = [road.road_id for road in route.roads] if route else None
            return self._waypoint_paths_at(wp_start, lookahead, road_ids)

        def _waypoint_paths_at(
            self,
            point: Sequence,
            lookahead: int,
            filter_road_ids: Sequence[str] = None,
        ) -> List[List[Waypoint]]:
            closest_linked_lp = (
                self._map._lanepoints.closest_linked_lanepoint_on_lane_to_point(
                    point, self._lane_id
                )
            )
            return self._map._waypoints_starting_at_lanepoint(
                closest_linked_lp,
                lookahead,
                tuple(filter_road_ids) if filter_road_ids else (),
                tuple(point),
            )

        def buffered_shape(self, width: float = 1.0) -> Polygon:
            return buffered_shape(self._sumo_lane.getShape(), width)

        @lru_cache(maxsize=8)
        def point_in_lane(self, point: Point) -> bool:
            # TAI:  could use (cached) self._sumo_lane.getBoundingBox(...) as a quick first-pass check...
            lane_point = self.to_lane_coord(point)
            return (
                abs(lane_point.t) <= self._width / 2 and 0 <= lane_point.s < self.length
            )

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

        def width_at_offset(self, offset: float) -> float:
            return self._width

        @lru_cache(maxsize=8)
        def from_lane_coord(self, lane_point: RefLinePoint) -> Point:
            shape = self._sumo_lane.getShape(False)
            x, y = sumolib.geomhelper.positionAtShapeOffset(shape, lane_point.s)
            return Point(x=x, y=y)

        @lru_cache(maxsize=8)
        def to_lane_coord(self, world_point: Point) -> RefLinePoint:
            return super().to_lane_coord(world_point)

        @lru_cache(maxsize=8)
        def center_at_point(self, point: Point) -> Point:
            return super().center_at_point(point)

        @lru_cache(8)
        def edges_at_point(self, point: Point) -> Tuple[Point, Point]:
            return super().edges_at_point(point)

        @lru_cache(8)
        def vector_at_offset(self, start_offset: float) -> np.ndarray:
            return super().vector_at_offset(start_offset)

        @lru_cache(maxsize=8)
        def center_pose_at_point(self, point: Point) -> Pose:
            return super().center_pose_at_point(point)

        @lru_cache(maxsize=8)
        def curvature_radius_at_offset(
            self, offset: float, lookahead: int = 5
        ) -> float:
            return super().curvature_radius_at_offset(offset, lookahead)

    def lane_by_id(self, lane_id: str) -> RoadMap.Lane:
        lane = self._lanes.get(lane_id)
        if lane:
            return lane
        sumo_lane = self._graph.getLane(lane_id)
        if not sumo_lane:
            self._log.warning(
                f"SumoRoadNetwork got request for unknown lane_id '{lane_id}'"
            )
            return None
        lane = SumoRoadNetwork.Lane(lane_id, sumo_lane, self)
        self._lanes[lane_id] = lane
        return lane

    class Road(RoadMap.Road):
        def __init__(self, road_id: str, sumo_edge: Edge, road_map):
            self._road_id = road_id
            self._sumo_edge = sumo_edge
            self._map = road_map

        @cached_property
        def is_junction(self) -> bool:
            return self._sumo_edge.isSpecial()

        @cached_property
        def length(self) -> float:
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

        @lru_cache(maxsize=16)
        def oncoming_roads_at_point(self, point: Point) -> List[RoadMap.Road]:
            result = []
            for lane in self.lanes:
                offset = lane.to_lane_coord(point).s
                result += [ol.road for ol in lane.oncoming_lanes_at_offset(offset)]
            return result

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

        @cached_property
        def lanes(self) -> List[RoadMap.Lane]:
            return [
                self._map.lane_by_id(sumo_lane.getID())
                for sumo_lane in self._sumo_edge.getLanes()
            ]

        def lane_at_index(self, index: int) -> RoadMap.Lane:
            return self.lanes[index]

        @lru_cache(maxsize=8)
        def point_on_road(self, point: Point) -> bool:
            # TAI:  could use (cached) self._sumo_edge.getBoundingBox(...) as a quick first-pass check...
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
            return buffered_shape(self._sumo_edge.getShape(), width)

    def road_by_id(self, road_id: str) -> RoadMap.Road:
        road = self._roads.get(road_id)
        if road:
            return road
        sumo_edge = self._graph.getEdge(road_id)
        if not sumo_edge:
            self._log.warning(
                f"SumoRoadNetwork got request for unknown road_id '{road_id}'"
            )
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
        # The includeJunctions parameter is the opposite of include_junctions because
        # what it does in the Sumo query is attach the "node" that is the junction (node)
        # shape to the shape of the non-special lanes that connect to it.  So if
        # includeJunctions is True, we are more likely to hit "normal" lanes
        # even when in an intersection where we want to hit "special"
        # lanes when we specify include_junctions=True.  Note that "special"
        # lanes are always candidates to be returned, no matter what.
        candidate_lanes = self._graph.getNeighboringLanes(
            point[0],
            point[1],
            r=radius,
            includeJunctions=not include_junctions,
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

    def generate_routes(
        self,
        start_road: RoadMap.Road,
        end_road: RoadMap.Road,
        via: Sequence[RoadMap.Road] = None,
        max_to_gen: int = 1,
    ) -> List[RoadMap.Route]:
        assert max_to_gen == 1, "multiple route generation not yet supported for Sumo"
        newroute = SumoRoadNetwork.Route(self)
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
            if len(sub_route) < 2:
                self._log.warning(
                    f"Unable to find valid path between {(cur_road.road_id, next_road.road_id)}."
                )
                return result
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
        route = SumoRoadNetwork.Route(self)
        next_edges = self._graph.getEdges(False)
        while next_edges and len(route.roads) < max_route_len:
            cur_edge = random.choice(next_edges)
            route.add_road(self.road_by_id(cur_edge.getID()))
            next_edges = list(cur_edge.getOutgoing().keys())
        return route

    def empty_route(self) -> RoadMap.Route:
        return SumoRoadNetwork.Route(self)

    def waypoint_paths(
        self,
        pose: Pose,
        lookahead: int,
        within_radius: float = 5,
        route: RoadMap.Route = None,
    ) -> List[List[Waypoint]]:
        if route:
            if route.roads:
                road_ids = [road.road_id for road in route.roads]
            else:
                road_ids = self._resolve_in_junction(pose)
            if road_ids:
                return self._waypoint_paths_along_route(
                    pose.position, lookahead, road_ids
                )
        closest_lps = self._lanepoints.closest_lanepoints(
            [pose], within_radius=within_radius
        )
        closest_lane = closest_lps[0].lane
        # TAI: the above lines could be replaced by:
        # closest_lane = self.nearest_lane(pose.position, radius=within_radius)
        waypoint_paths = []
        for lane in closest_lane.road.lanes:
            waypoint_paths += lane._waypoint_paths_at(pose.position, lookahead)
        return sorted(waypoint_paths, key=lambda p: p[0].lane_index)

    def _resolve_in_junction(self, pose: Pose) -> List[str]:
        # This is so that the waypoints don't jump between connections
        # when we don't know which lane we're on in a junction.
        # We take the 10 closest lanepoints then filter down to that which has
        # the closest heading. This way we get the lanepoint on our lane instead of
        # a potentially closer lane that is on a different junction connection.
        closest_lps = self._lanepoints.closest_lanepoints([pose], within_radius=None)
        closest_lps.sort(key=lambda lp: abs(pose.heading - lp.pose.heading))
        lane = closest_lps[0].lane
        if not lane.in_junction:
            return []
        road_ids = [lane.road.road_id]
        next_roads = lane.road.outgoing_roads
        assert (
            len(next_roads) <= 1
        ), "A junction is expected to have <= 1 outgoing roads"
        if next_roads:
            road_ids.append(next_roads[0].road_id)
        return road_ids

    def _waypoint_paths_along_route(
        self, point, lookahead: int, route: Sequence[str]
    ) -> List[List[Waypoint]]:
        """finds the closest lane to vehicle's position that is on its route,
        then gets waypoint paths from all lanes in its edge there."""
        assert len(route) > 0, f"Expected at least 1 road in the route, got: {route}"
        closest_llp_on_each_route_road = [
            self._lanepoints.closest_linked_lanepoint_on_road(point, road)
            for road in route
        ]
        closest_linked_lp = min(
            closest_llp_on_each_route_road,
            key=lambda l_lp: np.linalg.norm(
                vec_2d(l_lp.lp.pose.position) - vec_2d(point)
            ),
        )
        closest_lane = closest_linked_lp.lp.lane
        waypoint_paths = []
        for lane in closest_lane.road.lanes:
            waypoint_paths += lane._waypoint_paths_at(point, lookahead, route)
        return sorted(waypoint_paths, key=lambda p: p[0].lane_index)

    class Route(RoadMap.Route):
        def __init__(self, road_map):
            self._roads = []
            self._length = 0
            self._map = road_map

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

        @lru_cache(maxsize=8)
        def distance_between(self, start: Point, end: Point) -> float:
            route_roads = set(self._roads)
            for cand_start_lane, _ in self._map.nearest_lanes(start, 30.0, False):
                if cand_start_lane.road in route_roads:
                    break
            else:
                logging.warning("unable to find road on route near start point")
                return None
            start_road = cand_start_lane.road
            for cand_end_lane, _ in self._map.nearest_lanes(end, 30.0, False):
                if cand_end_lane.road in route_roads:
                    break
            else:
                logging.warning("unable to find road on route near end point")
                return None
            end_road = cand_end_lane.road
            d = 0
            for road in self._roads:
                if d == 0 and road == start_road:
                    start_offset = cand_start_lane.offset_along_lane(start)
                    d += cand_start_lane.length - start_offset
                elif road == end_road:
                    d += cand_end_lane.offset_along_lane(end)
                elif d > 0:
                    d += road.length
            return d

    def _compute_road_polygons(self):
        lane_to_poly = {}
        for edge in self._graph.getEdges():
            for lane in edge.getLanes():
                shape = buffered_shape(lane.getShape(), lane.getWidth())
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

    class _WaypointsCache:
        def __init__(self):
            self.lookahead = 0
            self.point = (0, 0, 0)
            self.filter_road_ids = ()
            self._starts = {}

        # XXX:  all vehicles share this cache now (as opposed to before
        # when it was in Plan.py and each vehicle had its own cache).
        # TODO: probably need to add vehicle_id to the key somehow (or just make it bigger)
        def _match(self, lookahead, point, filter_road_ids) -> bool:
            return (
                lookahead <= self.lookahead
                and point[0] == self.point[0]
                and point[1] == self.point[1]
                and filter_road_ids == self.filter_road_ids
            )

        def update(
            self,
            lookahead: int,
            point: Tuple[float, float, float],
            filter_road_ids: tuple,
            llp: LinkedLanePoint,
            paths: List[List[Waypoint]],
        ):
            if not self._match(lookahead, point, filter_road_ids):
                self.lookahead = lookahead
                self.point = point
                self.filter_road_ids = filter_road_ids
                self._starts = {}
            self._starts[llp.lp.lane.index] = paths

        def query(
            self,
            lookahead: int,
            point: Tuple[float, float, float],
            filter_road_ids: tuple,
            llp: LinkedLanePoint,
        ) -> List[List[Waypoint]]:
            if self._match(lookahead, point, filter_road_ids):
                hit = self._starts.get(llp.lp.lane.index, None)
                if hit:
                    # consider just returning all of them (not slicing)?
                    return [path[: (lookahead + 1)] for path in hit]
            return None

    def _waypoints_starting_at_lanepoint(
        self,
        lanepoint: LinkedLanePoint,
        lookahead: int,
        filter_road_ids: tuple,
        point: Tuple[float, float, float],
    ) -> List[List[Waypoint]]:
        """computes equally-spaced Waypoints for all lane paths starting at lanepoint
        up to lookahead waypoints ahead, constrained to filter_road_ids if specified."""

        # The following acts sort of like lru_cache(1), but it allows
        # for lookahead to be <= to the cached value...
        cache_paths = self._waypoints_cache.query(
            lookahead, point, filter_road_ids, lanepoint
        )
        if cache_paths:
            return cache_paths

        lanepoint_paths = self._lanepoints.paths_starting_at_lanepoint(
            lanepoint, lookahead, filter_road_ids
        )
        result = [
            SumoRoadNetwork._equally_spaced_path(path, point, self._lanepoints.spacing)
            for path in lanepoint_paths
        ]

        self._waypoints_cache.update(
            lookahead, point, filter_road_ids, lanepoint, result
        )

        return result

    @staticmethod
    def _equally_spaced_path(
        path: Sequence[LinkedLanePoint],
        point: Tuple[float, float, float],
        lp_spacing: float,
    ) -> List[Waypoint]:
        """given a list of LanePoints starting near point, that may not be evenly spaced,
        returns the same number of Waypoints that are evenly spaced and start at point."""

        continuous_variables = [
            "positions_x",
            "positions_y",
            "headings",
            "lane_width",
            "speed_limit",
        ]
        discrete_variables = ["lane_id", "lane_index"]

        ref_lanepoints_coordinates = {
            parameter: [] for parameter in (continuous_variables + discrete_variables)
        }
        for idx, lanepoint in enumerate(path):
            if lanepoint.is_inferred and 0 < idx < len(path) - 1:
                continue
            ref_lanepoints_coordinates["positions_x"].append(
                lanepoint.lp.pose.position[0]
            )
            ref_lanepoints_coordinates["positions_y"].append(
                lanepoint.lp.pose.position[1]
            )
            ref_lanepoints_coordinates["headings"].append(
                lanepoint.lp.pose.heading.as_bullet
            )
            ref_lanepoints_coordinates["lane_id"].append(lanepoint.lp.lane.lane_id)
            ref_lanepoints_coordinates["lane_index"].append(lanepoint.lp.lane.index)
            ref_lanepoints_coordinates["lane_width"].append(lanepoint.lp.lane._width)
            ref_lanepoints_coordinates["speed_limit"].append(
                lanepoint.lp.lane.speed_limit
            )

        ref_lanepoints_coordinates["headings"] = inplace_unwrap(
            ref_lanepoints_coordinates["headings"]
        )
        first_lp_heading = ref_lanepoints_coordinates["headings"][0]
        lp_position = path[0].lp.pose.position[:2]
        vehicle_pos = np.array(point[:2])
        heading_vec = np.array(radians_to_vec(first_lp_heading))
        projected_distant_lp_vehicle = np.inner(
            (vehicle_pos - lp_position), heading_vec
        )

        ref_lanepoints_coordinates["positions_x"][0] = (
            lp_position[0] + projected_distant_lp_vehicle * heading_vec[0]
        )
        ref_lanepoints_coordinates["positions_y"][0] = (
            lp_position[1] + projected_distant_lp_vehicle * heading_vec[1]
        )
        # To ensure that the distance between waypoints are equal, we used
        # interpolation approach inspired by:
        # https://stackoverflow.com/a/51515357
        cumulative_path_dist = np.cumsum(
            np.sqrt(
                np.ediff1d(ref_lanepoints_coordinates["positions_x"], to_begin=0) ** 2
                + np.ediff1d(ref_lanepoints_coordinates["positions_y"], to_begin=0) ** 2
            )
        )

        if len(cumulative_path_dist) <= lp_spacing:
            lp = path[0].lp
            return [
                Waypoint(
                    pos=lp.pose.position,
                    heading=lp.pose.heading,
                    lane_width=lp.lane._width,
                    speed_limit=lp.lane.speed_limit,
                    lane_id=lp.lane.lane_id,
                    lane_index=lp.lane.index,
                )
            ]

        evenly_spaced_cumulative_path_dist = np.linspace(
            0, cumulative_path_dist[-1], len(path)
        )

        evenly_spaced_coordinates = {}
        for variable in continuous_variables:
            evenly_spaced_coordinates[variable] = np.interp(
                evenly_spaced_cumulative_path_dist,
                cumulative_path_dist,
                ref_lanepoints_coordinates[variable],
            )

        for variable in discrete_variables:
            ref_coordinates = ref_lanepoints_coordinates[variable]
            evenly_spaced_coordinates[variable] = []
            jdx = 0
            for idx in range(len(path)):
                while (
                    jdx + 1 < len(cumulative_path_dist)
                    and evenly_spaced_cumulative_path_dist[idx]
                    > cumulative_path_dist[jdx + 1]
                ):
                    jdx += 1

                evenly_spaced_coordinates[variable].append(ref_coordinates[jdx])
            evenly_spaced_coordinates[variable].append(ref_coordinates[-1])

        equally_spaced_path = []
        for idx in range(len(path)):
            equally_spaced_path.append(
                Waypoint(
                    pos=np.array(
                        [
                            evenly_spaced_coordinates["positions_x"][idx],
                            evenly_spaced_coordinates["positions_y"][idx],
                        ]
                    ),
                    heading=Heading(evenly_spaced_coordinates["headings"][idx]),
                    lane_width=evenly_spaced_coordinates["lane_width"][idx],
                    speed_limit=evenly_spaced_coordinates["speed_limit"][idx],
                    lane_id=evenly_spaced_coordinates["lane_id"][idx],
                    lane_index=evenly_spaced_coordinates["lane_index"][idx],
                )
            )

        return equally_spaced_path
