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
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import List, NamedTuple, Sequence

import numpy as np

import warnings

with warnings.catch_warnings():
    from smarts.core.utils.file import suppress_pkg_resources

    warnings.filterwarnings("ignore", "numpy.ufunc size changed")
    with suppress_pkg_resources():
        # We force sklearn to use a different importer since sklearn's use of pkg_resources is
        # aggressive
        from sklearn.neighbors import KDTree

from smarts.core.utils.sumo import sumolib  # isort:skip
from sumolib.net.edge import Edge  # isort:skip
from sumolib.net.lane import Lane  # isort:skip

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

    def dist_to(self, p) -> float:
        """Calculates straight line distance to the given 2D point"""
        return np.linalg.norm(self.pos - vec_2d(p))

    def relative_heading(self, h: Heading) -> Heading:
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

    def signed_lateral_error(self, p) -> float:
        """Returns the signed lateral distance from the given point to the
        line formed by the lanepoint position and the lanepoints heading.

        Negative signals right of line and Positive left of line
        """
        return signed_dist_to_line(p, self.pos, self.heading.direction_vector())

    @property
    def pose(self) -> Pose:
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
        return hash((self.lp, tuple(nlp.lp for nlp in self.nexts)))


class LanePoints:
    def __init__(self, road_network, spacing: float, debug: bool = True):
        self.spacing = spacing
        self._road_network = road_network

        # XXX: for a big map, may not want to cache ALL of the potential LanePoints
        #      nor waste time here finding all of them.
        #      Lanepoints might be generated on demand based upon edges and lookahead.
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
    def _build_kd_tree(linked_lps: Sequence[LinkedLanePoint]) -> KDTree:
        return KDTree(np.array([l_lp.lp.pos for l_lp in linked_lps]), leaf_size=50)

    @staticmethod
    def _interpolate_shape_lanepoints(
        shape_lanepoints: Sequence[LinkedLanePoint], spacing: float
    ) -> List[LinkedLanePoint]:
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
    def _interpolate_from_shape_lp(
        shape_lp: LinkedLanePoint, spacing: float, interp_memo: dict
    ) -> Tuple[LinkedLanePoint, List[LinkedLanePoint]]:
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
        shape_lp: LinkedLanePoint,
        first_linked_lanepoint: LinkedLanePoint,
        next_shape_lp: LinkedLanePoint,
        spacing: float,
        newly_created_lanepoints: List[LinkedLanePoint],
    ) -> LinkedLanePoint:
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

    def _shape_lanepoints(self) -> List[LinkedLanePoint]:
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

    def _shape_lanepoints_along_lane(
        self, lane: Lane, lanepoint_by_lane_memo: dict
    ) -> Tuple[LinkedLanePoint, List[LinkedLanePoint]]:
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

    def _edge(self, lp: LanePoint) -> Edge:
        lane = self._road_network.lane_by_id(lp.lane_id)
        return lane.getEdge()

    @staticmethod
    def _closest_linked_lp_in_kd_tree_with_pose_batched(
        poses,
        lanepoints,
        tree,
        within_radius: float,
        k: int = 10,
        keep_all_k: bool = False,
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
    def _closest_linked_lp_in_kd_tree_batched(
        points, linked_lps, tree: KDTree, k: int = 1
    ):
        p2ds = np.array([vec_2d(p) for p in points])
        closest_indices = tree.query(
            p2ds, k=min(k, len(linked_lps)), return_distance=False, sort_results=True
        )
        return [[linked_lps[idx] for idx in idxs] for idxs in closest_indices]

    # TODO:  cleanup / consolidate these "closest_lanepoint*" methods (lots of redundancies)...

    def closest_lanepoint_batched(
        self, poses, filter_from_count: int, within_radius: float
    ) -> List[LanePoint]:
        linked_lanepoints = LanePoints._closest_linked_lp_in_kd_tree_with_pose_batched(
            poses,
            self._linked_lanepoints,
            self._lanepoints_kd_tree,
            within_radius=within_radius,
            k=filter_from_count,
        )
        return [l_lps[0].lp for l_lps in linked_lanepoints]

    def closest_lanepoint(
        self, pose: Pose, filter_from_count: int = 10, within_radius: float = 10
    ) -> LanePoint:
        return self.closest_lanepoint_batched(
            [pose], filter_from_count, within_radius=within_radius
        )[0]

    def closest_lanepoints(self, point, desired_count: int) -> List[LanePoint]:
        linked_lanepoints = LanePoints._closest_linked_lp_in_kd_tree_with_pose_batched(
            [point],
            self._linked_lanepoints,
            self._lanepoints_kd_tree,
            within_radius=10,
            k=desired_count,
            keep_all_k=True,
        )[0]
        return [llp.lp for llp in linked_lanepoints]

    def closest_lanepoint_on_lane(self, pose, lane_id: str) -> LanePoint:
        lane_kd_tree = self._lanepoints_kd_tree_by_lane_id[lane_id]
        linked_lanepoint = LanePoints._closest_linked_lp_in_kd_tree_with_pose_batched(
            [pose],
            self._lanepoints_by_lane_id[lane_id],
            lane_kd_tree,
            within_radius=10,
            k=10,
        )[0][0]
        return linked_lanepoint.lp

    def closest_lanepoint_on_lane_to_point(self, point, lane_id: str) -> LanePoint:
        return self.closest_linked_lanepoint_on_lane_to_point(point, lane_id).lp

    def closest_linked_lanepoint_on_lane_to_point(
        self, point, lane_id: str
    ) -> LinkedLanePoint:
        lane_kd_tree = self._lanepoints_kd_tree_by_lane_id[lane_id]
        return LanePoints._closest_linked_lp_in_kd_tree_batched(
            [point], self._lanepoints_by_lane_id[lane_id], lane_kd_tree, k=1
        )[0][0]

    def closest_linked_lanepoint_on_edge(self, point, edge_id: str) -> LinkedLanePoint:
        return LanePoints._closest_linked_lp_in_kd_tree_batched(
            [point],
            self._lanepoints_by_edge_id[edge_id],
            self._lanepoints_kd_tree_by_edge_id[edge_id],
        )[0][0]

    @lru_cache(maxsize=32)
    def paths_starting_at_lanepoint(
        self, lanepoint: LinkedLanePoint, lookahead: int, filter_edge_ids: tuple
    ) -> List[List[LinkedLanePoint]]:
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
