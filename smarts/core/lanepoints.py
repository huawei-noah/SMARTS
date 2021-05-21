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
from __future__ import (  # to allow for typing to refer to class being defined (LinkedLanePoint)
    annotations,
)

import math
import queue
import random
import warnings
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import List, NamedTuple, Sequence

import numpy as np

from smarts.core.utils.file import suppress_pkg_resources

from .utils.math import inplace_unwrap

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", "numpy.ufunc size changed")
    with suppress_pkg_resources():
        # We force sklearn to use a different importer since sklearn's use of pkg_resources is
        # aggressive
        from sklearn.neighbors import KDTree

from smarts.core.coordinates import Heading, Pose
from smarts.core.utils.math import (
    lerp,
    radians_to_vec,
    signed_dist_to_line,
    squared_dist,
    vec_2d,
    vec_to_radians,
)


@dataclass(frozen=True)
class LanePoint:
    """a.k.a. MapPoint:  static, based only on a map, always in a Lane"""

    pos: np.ndarray  # Point positioned on center of lane
    heading: Heading  # Heading angle of lane at this point (radians)
    lane_width: float  # Width of lane at this point (meters)
    speed_limit: float  # Lane speed in m/s
    lane_id: str  # ID of lane under lanepoint
    lane_index: int  # Index of the lane this lanepoint is over. 0 is the outer(right) most lane.

    def dist_to(self, p):
        """Calculates straight line distance to the given 2D point"""
        return np.linalg.norm(self.pos - vec_2d(p))

    def relative_heading(self, h: Heading):
        """Computes relative heading between the given angle and the lanepoint heading

        Returns:
            relative_heading: [-pi..pi]
        """
        assert isinstance(
            h, Heading
        ), "Heading h ({}) must be an instance of smarts.core.coordinates.Heading".format(
            type(h)
        )

        return self.heading.relative_to(h)

    def signed_lateral_error(self, p):
        """Returns the signed lateral distance from the given point to the
        line formed by the lanepoint position and the lanepoints heading.

        Negative signals right of line and Positive left of line
        """
        return signed_dist_to_line(p, self.pos, self.heading.direction_vector())

    @property
    def pose(self):
        return Pose.from_center(tuple(self.pos), self.heading)

    def __hash__(self):
        return hash(
            (
                *self.pos,
                self.heading,
                self.lane_width,
                self.speed_limit,
                self.lane_id,
                self.lane_index,
            )
        )

    def __eq__(self, other):
        if not isinstance(other, LanePoint):
            return False
        return (
            (self.pos == other.pos).all()
            and self.heading == other.heading
            and self.lane_width == other.lane_width
            and self.speed_limit == other.speed_limit
            and self.lane_id == other.lane_id
            and self.lane_index == other.lane_index
        )


class LinkedLanePoint(NamedTuple):
    lp: LanePoint = None
    is_shape_lp: bool = False
    nexts: List[LinkedLanePoint] = []  # list of next immediate LanePoint(s)
    # it's a list of LanePoints because a path may branch at junctions

    def __hash__(self):
        ## distinguish between different continuations here too
        ## so the lru_cache on _lanepoint_paths_starting_at_lanepoint() below
        ## doesn't return the wrong set of LanePoints.
        return hash(self.lp) + sum(hash(nlp.lp) for nlp in self.nexts)


@dataclass(frozen=True)
class Waypoint(LanePoint):
    """Dynamic, based on map and vehicle.  Unlike LanePoints,
    Waypoints do not have to snap to the middle of a road-network Lane,
    but rather they start just in front of a vehicle's present location.
    These are usually what is returned through a vehicle's sensors.
    Waypoints are always evenly-spaced."""

    @classmethod
    def from_LanePoint(cls, lp):
        return cls(
            pos=lp.pos,
            heading=lp.heading,
            lane_width=lp.lane_width,
            speed_limit=lp.speed_limit,
            lane_id=lp.lane_id,
            lane_index=lp.lane_index,
        )

    def __eq__(self, other):
        return super().__eq__(other)

    def __hash__(self):
        return super().__hash__()


class LanePoints:
    def __init__(self, road_network, spacing, debug=True):
        self.spacing = spacing
        self._next_lanepoint_id = 0
        self._road_network = road_network

        shape_lps = self._shape_lanepoints()
        self._linked_lanepoints = LanePoints._interpolate_shape_lanepoints(
            shape_lps, spacing
        )

        self._lanepoints_kd_tree = LanePoints._build_kd_tree(self._linked_lanepoints)

        self._lanepoints_by_lane_id = defaultdict(list)
        self._lanepoints_by_edge_id = defaultdict(list)
        for linked_lp in self._linked_lanepoints:
            lp_edge_id = self._edge(linked_lp.lp).getID()
            self._lanepoints_by_lane_id[linked_lp.lp.lane_id].append(linked_lp)
            self._lanepoints_by_edge_id[lp_edge_id].append(linked_lp)

        self._lanepoints_kd_tree_by_lane_id = {
            lane_id: LanePoints._build_kd_tree(l_lps)
            for lane_id, l_lps in self._lanepoints_by_lane_id.items()
        }

        self._lanepoints_kd_tree_by_edge_id = {
            edge_id: LanePoints._build_kd_tree(l_lps)
            for edge_id, l_lps in self._lanepoints_by_edge_id.items()
        }

    @staticmethod
    def _build_kd_tree(linked_lps):
        return KDTree(np.array([l_lp.lp.pos for l_lp in linked_lps]), leaf_size=50)

    @staticmethod
    def _interpolate_shape_lanepoints(shape_lanepoints, spacing):
        # memoize interpolated lanepoints on the shape lanepoint at start of
        # the line we are interpolating
        interp_memo = {}

        linked_lanepoints = []
        for shape_lp in shape_lanepoints:
            _, new_lanepoints = LanePoints._interpolate_from_shape_lp(
                shape_lp, spacing, interp_memo
            )
            linked_lanepoints += new_lanepoints

        return linked_lanepoints

    @staticmethod
    def _interpolate_from_shape_lp(shape_lp, spacing, interp_memo):
        shape_queue = queue.Queue()
        shape_queue.put((shape_lp, None))
        newly_created_lanepoints = []
        initial_lanepoint = None

        while not shape_queue.empty():
            shape_lp, previous_lp = shape_queue.get()
            first_linked_lanepoint = interp_memo.get(shape_lp.lp)
            if first_linked_lanepoint:
                if previous_lp:
                    previous_lp.nexts.append(first_linked_lanepoint)
                continue

            first_linked_lanepoint = LinkedLanePoint(
                lp=shape_lp.lp,  # lanepoints are frozen, so no need to copy lp here
                nexts=[],
                is_shape_lp=True,
            )

            if previous_lp is not None:
                previous_lp.nexts.append(first_linked_lanepoint)

            if initial_lanepoint is None:
                initial_lanepoint = first_linked_lanepoint

            interp_memo[shape_lp.lp] = first_linked_lanepoint

            newly_created_lanepoints.append(first_linked_lanepoint)

            for current_shape_lp in shape_lp.nexts:
                next_shape_lp = LanePoints._process_interp_for_lane_lp(
                    shape_lp,
                    first_linked_lanepoint,
                    current_shape_lp,
                    spacing,
                    newly_created_lanepoints,
                )

                shape_queue.put((current_shape_lp, next_shape_lp))

        return initial_lanepoint, newly_created_lanepoints

    @staticmethod
    def _process_interp_for_lane_lp(
        shape_lp,
        first_linked_lanepoint,
        next_shape_lp,
        spacing,
        newly_created_lanepoints,
    ):
        lane_id = shape_lp.lp.lane_id

        curr_lanepoint = first_linked_lanepoint

        lane_seg_vec = next_shape_lp.lp.pos - shape_lp.lp.pos
        lane_seg_len = np.linalg.norm(lane_seg_vec)

        # We set the initial distance into the lane at `spacing` because
        # we already have a lanepoint along this segment (curr_lanepoint)
        dist_into_lane_seg = spacing
        while dist_into_lane_seg < lane_seg_len:
            p = dist_into_lane_seg / lane_seg_len
            pos = shape_lp.lp.pos + lane_seg_vec * p

            # The thresholds for calculating last lanepoint. If the
            # midpoint between the current lanepoint and the next shape
            # lanepoint is less than the minimum distance then the last
            # lanepoint position will be that midpoint. If the midpoint
            # is closer than last spacing threshold to the next shape
            # lanepoint, then the last lanepoint will be the current
            # lanepoint.
            # XXX: the map scale should be taken into account here.
            last_spacing_threshold_dist = 0.8 * spacing
            minimum_dist_next_shape_lp = 1.4

            half_distant_current_next_shape_lp = np.linalg.norm(
                0.5 * (curr_lanepoint.lp.pos - next_shape_lp.lp.pos)
            )
            mid_point_current_next_shape_lp = 0.5 * (
                next_shape_lp.lp.pos + curr_lanepoint.lp.pos
            )
            if half_distant_current_next_shape_lp < minimum_dist_next_shape_lp:
                pos = mid_point_current_next_shape_lp
            dist_pos_next_shape_lp = np.linalg.norm(next_shape_lp.lp.pos - pos)
            if dist_pos_next_shape_lp < last_spacing_threshold_dist:
                break

            heading = vec_to_radians(lane_seg_vec)
            lane_width = lerp(shape_lp.lp.lane_width, next_shape_lp.lp.lane_width, p)
            speed_limit = lerp(shape_lp.lp.speed_limit, next_shape_lp.lp.speed_limit, p)

            linked_lanepoint = LinkedLanePoint(
                lp=LanePoint(
                    pos=pos,
                    heading=Heading(heading),
                    lane_width=lane_width,
                    speed_limit=speed_limit,
                    lane_id=lane_id,
                    lane_index=shape_lp.lp.lane_index,
                ),
                nexts=[],
                is_shape_lp=False,
            )

            curr_lanepoint.nexts.append(linked_lanepoint)
            curr_lanepoint = linked_lanepoint
            newly_created_lanepoints.append(linked_lanepoint)
            dist_into_lane_seg += spacing
        return curr_lanepoint

    def _shape_lanepoints(self):
        """Computes the lane shape (start/shape/end) lanepoints for all lanes in
        the network, the result of this function can be used to interpolate
        lanepoints along lanes to the desired granularity.
        """
        # Don't request internal lanes since we get them by calling
        # `lane.getViaLaneID()`
        edges = self._road_network.graph.getEdges(False)
        lanepoint_by_lane_memo = {}
        shape_lanepoints = []

        for edge in edges:
            for lane in edge.getLanes():
                _, new_lps = self._shape_lanepoints_along_lane(
                    lane, lanepoint_by_lane_memo
                )
                shape_lanepoints += new_lps

        return shape_lanepoints

    def _shape_lanepoints_along_lane(self, lane, lanepoint_by_lane_memo):
        lane_queue = queue.Queue()
        lane_queue.put((lane, None))
        shape_lanepoints = []
        initial_lanepoint = None
        while not lane_queue.empty():
            lane, previous_lp = lane_queue.get()
            first_lanepoint = lanepoint_by_lane_memo.get(lane.getID())
            if first_lanepoint:
                if previous_lp:
                    previous_lp.nexts.append(first_lanepoint)
                continue

            lane_data = self._road_network.lane_data_for_lane(lane)
            lane_shape = [np.array(p) for p in lane.getShape(False)]

            assert len(lane_shape) >= 2, repr(lane_shape)

            heading = vec_to_radians(lane_shape[1] - lane_shape[0])
            heading = Heading(heading)

            first_lanepoint = LinkedLanePoint(
                lp=LanePoint(
                    pos=lane_shape[0],
                    heading=heading,
                    lane_width=lane.getWidth(),
                    speed_limit=lane_data.lane_speed,
                    lane_id=lane.getID(),
                    lane_index=lane.getIndex(),
                ),
                nexts=[],
                is_shape_lp=True,
            )

            if previous_lp is not None:
                previous_lp.nexts.append(first_lanepoint)

            if initial_lanepoint is None:
                initial_lanepoint = first_lanepoint

            lanepoint_by_lane_memo[lane.getID()] = first_lanepoint
            shape_lanepoints.append(first_lanepoint)
            curr_lanepoint = first_lanepoint

            for p1, p2 in zip(lane_shape[1:], lane_shape[2:]):
                heading_ = vec_to_radians(p2 - p1)
                heading_ = Heading(heading_)
                linked_lanepoint = LinkedLanePoint(
                    lp=LanePoint(
                        pos=p1,
                        heading=heading_,
                        lane_width=lane.getWidth(),
                        speed_limit=lane_data.lane_speed,
                        lane_id=lane.getID(),
                        lane_index=lane.getIndex(),
                    ),
                    nexts=[],
                    is_shape_lp=True,
                )

                shape_lanepoints.append(linked_lanepoint)
                curr_lanepoint.nexts.append(linked_lanepoint)
                curr_lanepoint = linked_lanepoint

            # Add a lanepoint for the last point of the current lane
            last_linked_lanepoint = LinkedLanePoint(
                lp=LanePoint(
                    pos=lane_shape[-1],
                    heading=curr_lanepoint.lp.heading,
                    lane_width=curr_lanepoint.lp.lane_width,
                    speed_limit=curr_lanepoint.lp.speed_limit,
                    lane_id=curr_lanepoint.lp.lane_id,
                    lane_index=curr_lanepoint.lp.lane_index,
                ),
                nexts=[],
                is_shape_lp=True,
            )

            shape_lanepoints.append(last_linked_lanepoint)
            curr_lanepoint.nexts.append(last_linked_lanepoint)
            curr_lanepoint = last_linked_lanepoint

            for out_connection in lane.getOutgoing():
                out_lane = out_connection.getToLane()

                # Use internal lanes of junctions (if we're at a junction)
                via_lane_id = out_connection.getViaLaneID()
                if via_lane_id:
                    out_lane = self._road_network.lane_by_id(via_lane_id)

                lane_queue.put((out_lane, curr_lanepoint))

        return initial_lanepoint, shape_lanepoints

    def _edge(self, lp):
        lane = self._road_network.lane_by_id(lp.lane_id)
        return lane.getEdge()

    @staticmethod
    def _closest_linked_lp_in_kd_tree_with_pose_batched(
        poses, lanepoints, tree, within_radius, k=10, keep_all_k=False
    ):
        linked_lanepoints = LanePoints._closest_linked_lp_in_kd_tree_batched(
            [pose.position[:2] for pose in poses], lanepoints, tree, k=k
        )

        radius_sq = within_radius * within_radius
        linked_lanepoints = [
            sorted(
                l_lps,
                key=lambda _llp: squared_dist(poses[idx].position[:2], _llp.lp.pos),
            )
            for idx, l_lps in enumerate(linked_lanepoints)
        ]
        # exclude those outside radius except closest
        if not keep_all_k:
            linked_lanepoints = [
                [
                    _llp
                    for i, _llp in enumerate(_llps)
                    if squared_dist(poses[idx].position[:2], _llp.lp.pos) <= radius_sq
                    or i == 0
                ]
                for idx, _llps in enumerate(linked_lanepoints)
            ]
        # Get the nearest point for the points where the radius check failed
        unfound_lanepoints = [
            (i, poses[i])
            for i, group in enumerate(linked_lanepoints)
            if len(group) == 0
        ]
        if len(unfound_lanepoints) > 0:
            remaining_linked_lps = LanePoints._closest_linked_lp_in_kd_tree_batched(
                [pose.position[:2] for _, pose in unfound_lanepoints],
                lanepoints,
                tree=tree,
                k=k,
            )
            # Replace the empty lanepoint locations
            for (i, _), lps in [
                g for g in zip(unfound_lanepoints, remaining_linked_lps)
            ]:
                linked_lanepoints[i] = [lps]

        return [
            sorted(
                l_lps,
                key=lambda _llp: squared_dist(poses[idx].position[:2], _llp.lp.pos)
                + abs(poses[idx].heading.relative_to(_llp.lp.heading)),
            )
            for idx, l_lps in enumerate(linked_lanepoints)
        ]

    @staticmethod
    def _closest_linked_lp_in_kd_tree_batched(points, linked_lps, tree, k=1):
        p2ds = np.array([vec_2d(p) for p in points])
        closest_indices = tree.query(
            p2ds, k=min(k, len(linked_lps)), return_distance=False, sort_results=True
        )

        return [[linked_lps[idx] for idx in idxs] for idxs in closest_indices]

    def closest_lanepoint_batched(self, poses, filter_from_count, within_radius):
        linked_lanepoints = LanePoints._closest_linked_lp_in_kd_tree_with_pose_batched(
            poses,
            self._linked_lanepoints,
            self._lanepoints_kd_tree,
            within_radius=within_radius,
            k=filter_from_count,
        )

        return [l_lps[0].lp for l_lps in linked_lanepoints]

    def closest_lanepoint(self, pose, filter_from_count=10, within_radius=10):
        return self.closest_lanepoint_batched(
            [pose], filter_from_count, within_radius=within_radius
        )[0]

    def closest_lanepoints(self, point, desired_count):
        linked_lanepoints = LanePoints._closest_linked_lp_in_kd_tree_with_pose_batched(
            [point],
            self._linked_lanepoints,
            self._lanepoints_kd_tree,
            within_radius=10,
            k=desired_count,
            keep_all_k=True,
        )[0]

        return [llp.lp for llp in linked_lanepoints]

    def closest_lanepoint_on_lane(self, pose, lane_id):
        lane_kd_tree = self._lanepoints_kd_tree_by_lane_id[lane_id]
        linked_lanepoint = LanePoints._closest_linked_lp_in_kd_tree_with_pose_batched(
            [pose],
            self._lanepoints_by_lane_id[lane_id],
            lane_kd_tree,
            within_radius=10,
            k=10,
        )[0][0]

        return linked_lanepoint.lp

    def closest_lanepoint_on_lane_to_point(self, position, lane_id):
        lane_kd_tree = self._lanepoints_kd_tree_by_lane_id[lane_id]
        linked_lanepoint = LanePoints._closest_linked_lp_in_kd_tree_batched(
            [position], self._lanepoints_by_lane_id[lane_id], lane_kd_tree
        )[0][0]

        return linked_lanepoint.lp

    ## STEVE TODO:  everything from here down should be moved to MissionPlanner

    def waypoint_paths_on_lane_at(
        self, point, lane_id, lookahead, filter_edge_ids: Sequence[str] = None
    ):
        lane_kd_tree = self._lanepoints_kd_tree_by_lane_id[lane_id]
        closest_linked_lp = LanePoints._closest_linked_lp_in_kd_tree_batched(
            [point], self._lanepoints_by_lane_id[lane_id], lane_kd_tree, k=1
        )[0][0]
        return self._waypoints_starting_at_lanepoint(
            closest_linked_lp,
            lookahead,
            tuple(filter_edge_ids) if filter_edge_ids else (),
            tuple(point),
        )

    def waypoint_paths_at(self, pose, lookahead, filter_from_count=3, within_radius=5):
        closest_linked_lp = self.closest_lanepoint(
            pose, filter_from_count=filter_from_count, within_radius=within_radius
        )
        closest_lane = self._road_network.lane_by_id(closest_linked_lp.lane_id)

        waypoint_paths = []
        for lane in closest_lane.getEdge().getLanes():
            lane_id = lane.getID()
            waypoint_paths += self.waypoint_paths_on_lane_at(
                pose.position, lane_id, lookahead
            )

        sorted_wps = sorted(waypoint_paths, key=lambda p: p[0].lane_index)
        return sorted_wps

    def waypoint_paths_along_route(self, point, lookahead, route):
        assert len(route) > 0, f"Expected at least 1 edge in the route, got: {route}"
        closest_lp_on_each_route_edge = [
            LanePoints._closest_linked_lp_in_kd_tree_batched(
                [point],
                self._lanepoints_by_edge_id[edge],
                self._lanepoints_kd_tree_by_edge_id[edge],
            )[0][0]
            for edge in route
        ]

        closest_linked_lp = min(
            closest_lp_on_each_route_edge, key=lambda l_lp: l_lp.lp.dist_to(point)
        )
        closest_lane = self._road_network.lane_by_id(closest_linked_lp.lp.lane_id)

        waypoint_paths = []
        for lane in closest_lane.getEdge().getLanes():
            lane_id = lane.getID()
            waypoint_paths += self.waypoint_paths_on_lane_at(
                point, lane_id, lookahead, route
            )

        sorted_wps = sorted(waypoint_paths, key=lambda p: p[0].lane_index)
        return sorted_wps

    def _waypoints_starting_at_lanepoint(
        self, lanepoint: LinkedLanePoint, lookahead: int, filter_edge_ids: tuple, point
    ):
        lanepoint_paths = self._lanepoint_paths_starting_at_lanepoint(
            lanepoint, lookahead, filter_edge_ids
        )
        return [self._equally_spaced_path(path, point) for path in lanepoint_paths]

    @lru_cache(maxsize=32)
    def _lanepoint_paths_starting_at_lanepoint(
        self, lanepoint: LinkedLanePoint, lookahead: int, filter_edge_ids: tuple
    ):
        lanepoint_paths = [[lanepoint]]
        for _ in range(lookahead):
            next_lanepoint_paths = []
            for path in lanepoint_paths:
                branching_paths = []
                for next_lp in path[-1].nexts:
                    # TODO: This could be a problem. What about internal lanes?
                    # Filter only the edges we're interested in
                    edge_id = self._edge(next_lp.lp).getID()
                    if filter_edge_ids and edge_id not in filter_edge_ids:
                        continue
                    new_path = path + [next_lp]
                    branching_paths.append(new_path)

                if branching_paths == []:
                    branching_paths = [path]

                next_lanepoint_paths += branching_paths

            lanepoint_paths = next_lanepoint_paths

        return lanepoint_paths

    def _equally_spaced_path(self, path, point):
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
            if not lanepoint.is_shape_lp and 0 < idx < len(path) - 1:
                continue
            ref_lanepoints_coordinates["positions_x"].append(lanepoint.lp.pos[0])
            ref_lanepoints_coordinates["positions_y"].append(lanepoint.lp.pos[1])
            ref_lanepoints_coordinates["headings"].append(
                lanepoint.lp.heading.as_bullet
            )
            ref_lanepoints_coordinates["lane_id"].append(lanepoint.lp.lane_id)
            ref_lanepoints_coordinates["lane_index"].append(lanepoint.lp.lane_index)
            ref_lanepoints_coordinates["lane_width"].append(lanepoint.lp.lane_width)
            ref_lanepoints_coordinates["speed_limit"].append(lanepoint.lp.speed_limit)

        ref_lanepoints_coordinates["headings"] = inplace_unwrap(
            ref_lanepoints_coordinates["headings"]
        )
        first_lp_heading = ref_lanepoints_coordinates["headings"][0]
        lp_position = np.array([*path[0].lp.pos, 0])
        vehicle_pos = np.array([point[0], point[1], 0])
        heading_vector = np.array(
            [
                *radians_to_vec(first_lp_heading),
                0,
            ]
        )
        projected_distant_lp_vehicle = np.inner(
            (vehicle_pos - lp_position), heading_vector
        )

        ref_lanepoints_coordinates["positions_x"][0] = (
            lp_position[0] + projected_distant_lp_vehicle * heading_vector[0]
        )
        ref_lanepoints_coordinates["positions_y"][0] = (
            lp_position[1] + projected_distant_lp_vehicle * heading_vector[1]
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
            return [Waypoint.from_LanePoint(path[0].lp)]

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
