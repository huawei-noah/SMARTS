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
from typing import List, Sequence, Tuple

import numpy as np

from .coordinates import Heading, Pose
from .planner import Mission, Planner, Waypoint
from .road_map import RoadMap
from .sumo_lanepoints import LanePoint, LinkedLanePoint
from .sumo_road_network import SumoRoadNetwork
from .utils.math import inplace_unwrap, radians_to_vec, vec_2d


class SumoPlanner(Planner):
    def __init__(self, road_map: RoadMap):
        super().__init__(road_map)
        assert isinstance(self._road_map, SumoRoadNetwork)
        self._log = logging.getLogger(self.__class__.__name__)
        self._lanepoints = self._road_map.lanepoints
        self._waypoints_cache = SumoPlanner._WaypointsCache()

    def waypoint_paths(
        self,
        pose: Pose,
        lookahead: int,
        within_radius: float = 5,
        constrain_to_route: bool = True,
    ) -> List[List[Waypoint]]:
        if constrain_to_route:
            assert self._route, "Must call plan(...) before being able to use a route."
            road_ids = self._road_ids(pose)
            if road_ids:
                return self._waypoint_paths_along_route(
                    pose.position, lookahead, road_ids
                )

        return self._waypoint_paths_at(pose, lookahead, within_radius)

    def waypoint_paths_on_lane_at_point(
        self, pose: Pose, lane_id: str, lookahead: int, constrain_to_route: bool = True
    ) -> List[List[Waypoint]]:
        road_ids = None
        if constrain_to_route:
            assert self._route, "Must call plan(...) before being able to use a route."
            road_ids = self._road_ids(pose, lane_id)

        return self._waypoint_paths_on_lane_at(
            pose.position, lane_id, lookahead, road_ids
        )

    def waypoint_paths_on_lane_at_offset(
        self, lane: RoadMap.Lane, offset: float, lookahead: int = 30
    ) -> List[List[Waypoint]]:
        wp_start = lane.from_lane_coord(RefLinePoint(offset))
        return self._waypoint_paths_on_lane_at(
            point=wp_start,
            lane_id=lane.getID(),
            lookahead=lookahead,
        )

    def _road_ids(self, pose: Pose, lane_id: str = None) -> List[str]:
        if self._mission.has_fixed_route:
            return [road.road_id for road in self._route.roads]

        # Filter lanepoints to the internal lane the vehicle is driving on to deal w/
        # non-fixed routes (e.g. endless missions). This is so that the lanepoints don't
        # jump between junction connections.
        if lane_id is None:
            # We take the 10 closest lanepoints to then filter down to that which has
            # the closest heading. This way we get the lanepoint on our lane instead of
            # a potentially closer lane that is on a different junction connection.
            closest_lps = self._lanepoints.closest_lanepoints(
                poses=[pose], within_radius=None, maximum_count=10
            )
            closest_lps = sorted(
                closest_lps, key=lambda lp: abs(pose.heading - lp.pose.heading)
            )
            lane_id = closest_lps[0].lane.lane_id

        lane = self._road_map.lane_by_id(lane_id)
        if not lane.in_junction:
            return []

        road = lane.road
        road_ids = [road.road_id]

        next_roads = road.outgoing_roads
        assert (
            len(next_roads) <= 1
        ), "A junction is expected to have <= 1 outgoing roads"
        if next_roads:
            road_ids.append(next_roads[0].road_id)

        return road_ids

    def _waypoint_paths_on_lane_at(
        self,
        point: Sequence,
        lane_id: str,
        lookahead: int,
        filter_road_ids: Sequence[str] = None,
    ) -> List[List[Waypoint]]:
        """computes equally-spaced Waypoints for all lane paths
        up to lookahead waypoints ahead, constrained to filter_road_ids if specified,
        starting at the nearest LanePoint to point within lane lane_id."""
        closest_linked_lp = self._lanepoints.closest_linked_lanepoint_on_lane_to_point(
            point, lane_id
        )
        return self._waypoints_starting_at_lanepoint(
            closest_linked_lp,
            lookahead,
            tuple(filter_road_ids) if filter_road_ids else (),
            tuple(point),
        )

    def _waypoint_paths_at(
        self,
        pose: Pose,
        lookahead: int,
        within_radius: int = 5,
    ) -> List[List[Waypoint]]:
        closest_lps = self._lanepoints.closest_lanepoints(
            [pose], within_radius=within_radius
        )
        closest_lane = closest_lps[0].lane
        # TAI: the above lines could be replaced by:
        # closest_lane = self._road_map.nearest_lane(pose.point, radius=within_radius)
        waypoint_paths = []
        for lane in closest_lane.road.lanes:
            waypoint_paths += self._waypoint_paths_on_lane_at(
                pose.position, lane.lane_id, lookahead
            )
        return sorted(waypoint_paths, key=lambda p: p[0].lane_index)

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
            waypoint_paths += self._waypoint_paths_on_lane_at(
                point, lane.lane_id, lookahead, route
            )
        return sorted(waypoint_paths, key=lambda p: p[0].lane_index)

    class _WaypointsCache:
        def __init__(self):
            self.lookahead = 0
            self.point = (0, 0, 0)
            self.filter_road_ids = ()
            self._starts = {}

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
            SumoPlanner._equally_spaced_path(path, point) for path in lanepoint_paths
        ]

        self._waypoints_cache.update(
            lookahead, point, filter_road_ids, lanepoint, result
        )

        return result

    @staticmethod
    def _equally_spaced_path(
        path: Sequence[LinkedLanePoint], point: Tuple[float, float, float]
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

        if len(cumulative_path_dist) <= 1:
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
