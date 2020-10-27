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
from typing import List, Sequence, Tuple
from numpy import unique
from smarts.core.sumo_road_network import SumoRoadNetwork

from .utils.sumo import sumolib
from sumolib.net.edge import Edge


class Route:
    """The Route interface"""

    @property
    def edges(self) -> List[Edge]:
        """An (unordered) list of edges that this route covers"""
        raise NotImplementedError()

    @property
    def geometry(self) -> Sequence[Sequence[Tuple[float, float]]]:
        """A sequence of polygons describing the shape of each edge on the route"""
        raise NotImplementedError()

    @property
    def length(self) -> float:
        """Compute the length of this route, if the route branches, the maximum length is calculated"""
        return 0


class EmptyRoute(Route):
    @property
    def edges(self) -> List[Edge]:
        return []

    @property
    def geometry(self) -> Sequence[Tuple[float, float]]:
        return []

    @property
    def length(self) -> float:
        return 0


class ShortestRoute(Route):
    def __init__(
        self,
        sumo_road_network: SumoRoadNetwork,
        edge_constraints: Sequence[Edge],
        wraps_around: bool = False,
    ):
        self.road_network = sumo_road_network
        self.wraps_around = wraps_around

        if len(edge_constraints) < 2:
            raise ValueError(
                "Must provide two edges or more to plan a route with "
                f"len(self.edge_constraints)={len(self.edge_constaints)}"
            )
        if len(set(edge_constraints)) == 1:
            # ie. starting and ending on the same edge
            edge_constraints = list(set(edge_constraints))

        self.route_without_internal_edges = self._fill_route_missing_edges(
            edge_constraints
        )

        self._cached_edges = self._compute_edges()
        self._cached_geometry = self._compute_geometry()
        self._cached_length = self._compute_length()

    @property
    def edges(self) -> List[Edge]:
        return self._cached_edges

    @property
    def geometry(self) -> Sequence[Tuple[float, float]]:
        return self._cached_geometry

    @property
    def length(self) -> float:
        return self._cached_length

    def _compute_edges(self):
        # Handle case when route is within a single edge
        if len(self.route_without_internal_edges) == 1:
            return self.route_without_internal_edges

        edges = list()
        edge_ids = list()
        adjacent_edge_pairs = zip(
            self.route_without_internal_edges, self.route_without_internal_edges[1:]
        )
        for curr_edge, next_edge in adjacent_edge_pairs:
            internal_routes = self._internal_routes_between(curr_edge, next_edge)
            for internal_route in internal_routes:
                edges.extend(internal_route)
                edge_ids.extend([edge.getID() for edge in internal_route])
        _, indices = unique(edge_ids, return_index=True)
        return [edges[idx] for idx in sorted(indices)]

    def _compute_geometry(self):
        return [
            SumoRoadNetwork.buffered_lane_or_edge(
                edge, width=sum([lane.getWidth() for lane in edge.getLanes()])
            )
            for edge in self.edges
        ]

    def _compute_length(self):
        route_length = 0
        adjacent_edge_pairs = zip(
            self.route_without_internal_edges, self.route_without_internal_edges[1:]
        )
        for curr_edge, next_edge in adjacent_edge_pairs:
            internal_routes = self._internal_routes_between(curr_edge, next_edge)

            longest_internal_route_length = max(
                sum(
                    max(lane.getLength() for lane in edge.getLanes())
                    for edge in internal_route
                )
                for internal_route in internal_routes
            )

            route_length += longest_internal_route_length

        return route_length

    def _fill_route_missing_edges(self, route):
        edges = []
        for curr_edge, next_edge in zip(route, route[1:] + [None]):
            # We're at the end of the edges
            if next_edge is None:
                edges.append(curr_edge)
                break

            sub_route = (
                self.road_network.graph.getShortestPath(curr_edge, next_edge)[0] or []
            )
            assert (
                len(sub_route) >= 2
            ), f"Unable to find valid path (len={len(sub_route)}) between {(curr_edge, next_edge)}"
            # The sub route includes the boundary edges (curr_edge, next_edge). We
            # clip the latter to prevent duplicates
            edges.extend(sub_route[:-1])

        return edges

    def _internal_routes_between(self, start_edge, end_edge):
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
                via_lane = self.road_network.lane_by_id(via_lane_id)
                via_edge = via_lane.getEdge()

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
