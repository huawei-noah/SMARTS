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

# to allow for typing to refer to class being defined (LinkedLanePoint)
from __future__ import annotations

import math
import queue
import warnings
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import List, NamedTuple, Sequence, Tuple
import numpy as np

with warnings.catch_warnings():
    from smarts.core.utils.file import suppress_pkg_resources

    warnings.filterwarnings("ignore", "numpy.ufunc size changed")
    with suppress_pkg_resources():
        # We force sklearn to use a different importer since sklearn's use of pkg_resources is
        # aggressive
        from sklearn.neighbors import KDTree

from smarts.core.coordinates import Heading, Pose
from smarts.core.road_map import RoadMap
from smarts.core.utils.math import (
    fast_quaternion_from_angle,
    lerp,
    squared_dist,
    vec_2d,
    vec_to_radians,
)


@dataclass(frozen=True)
class LanePoint:
    lane: RoadMap.Lane
    pose: Pose


class LinkedLanePoint(NamedTuple):
    lp: LanePoint = None
    is_inferred: bool = True
    nexts: List[LinkedLanePoint] = []  # list of next immediate LanePoint(s)
    # it's a list of LanePoints because a path may branch at junctions

    def __hash__(self):
        ## distinguish between different continuations here too
        ## so the lru_cache on _lanepoint_paths_starting_at_lanepoint() below
        ## doesn't return the wrong set of LanePoints.
        return hash((self.lp, tuple(nlp.lp for nlp in self.nexts)))


class OpenDriveLanePoints:
    def __init__(self, road_map: RoadMap, spacing: float, debug: bool = True):
        assert isinstance(road_map, OpenDriveRoadNetwork)
        self._road_map = road_map
        self.spacing = spacing

        # XXX: for a big map, may not want to cache ALL of the potential LanePoints
        #      nor waste time here finding all of them.
        #      Lanepoints might be generated on demand based upon edges and lookahead.
        shape_lps = self._shape_lanepoints()
        self._linked_lanepoints = OpenDriveLanePoints._interpolate_shape_lanepoints(
            shape_lps, spacing
        )

        self._lanepoints_kd_tree = OpenDriveLanePoints._build_kd_tree(
            self._linked_lanepoints
        )

        self._lanepoints_by_lane_id = defaultdict(list)
        self._lanepoints_by_road_id = defaultdict(list)
        for linked_lp in self._linked_lanepoints:
            lp_road_id = linked_lp.lp.lane.road.road_id
            self._lanepoints_by_lane_id[linked_lp.lp.lane.lane_id].append(linked_lp)
            self._lanepoints_by_road_id[lp_road_id].append(linked_lp)

        self._lanepoints_kd_tree_by_lane_id = {
            lane_id: OpenDriveLanePoints._build_kd_tree(l_lps)
            for lane_id, l_lps in self._lanepoints_by_lane_id.items()
        }

        self._lanepoints_kd_tree_by_road_id = {
            road_id: OpenDriveLanePoints._build_kd_tree(l_lps)
            for road_id, l_lps in self._lanepoints_by_road_id.items()
        }

    @staticmethod
    def _build_kd_tree(linked_lps: Sequence[LinkedLanePoint]) -> KDTree:
        return KDTree(
            np.array([l_lp.lp.pose.position for l_lp in linked_lps]), leaf_size=50
        )

    def _shape_lanepoints(self) -> List[LinkedLanePoint]:
        """Computes the lane shape (start/shape/end) lanepoints for all lanes in
        the network, the result of this function can be used to interpolate
        lanepoints along lanes to the desired granularity.
        """
        roads = self._road_map._roads
        lanepoint_by_lane_memo = {}
        shape_lanepoints = []

        for road_id in roads:
            road = roads[road_id]
            for lane in road.lanes:
                # Ignore lanes in OpenDRIVE not drivable
                if lane.is_drivable:
                    _, new_lps = self._shape_lanepoints_along_lane(
                        lane, lanepoint_by_lane_memo
                    )
                    shape_lanepoints += new_lps

        return shape_lanepoints

    def _shape_lanepoints_along_lane(
        self, lane: RoadMap.Lane, lanepoint_by_lane_memo: dict
    ) -> Tuple[LinkedLanePoint, List[LinkedLanePoint]]:
        lane_queue = queue.Queue()
        lane_queue.put((lane, None))
        shape_lanepoints = []
        initial_lanepoint = None
        while not lane_queue.empty():
            curr_lane, previous_lp = lane_queue.get()
            first_lanepoint = lanepoint_by_lane_memo.get(curr_lane.lane_id)
            if first_lanepoint:
                if previous_lp:
                    previous_lp.nexts.append(first_lanepoint)
                continue

            lane_shape = [np.array(p) for p in curr_lane.centerline_points]

            assert len(lane_shape) >= 2, repr(lane_shape)

            heading = vec_to_radians(lane_shape[1] - lane_shape[0])
            heading = Heading(heading)
            orientation = fast_quaternion_from_angle(heading)

            first_lanepoint = LinkedLanePoint(
                lp=LanePoint(
                    lane=curr_lane,
                    pose=Pose(position=lane_shape[0], orientation=orientation),
                ),
                nexts=[],
                is_inferred=False,
            )

            if previous_lp is not None:
                previous_lp.nexts.append(first_lanepoint)

            if initial_lanepoint is None:
                initial_lanepoint = first_lanepoint

            lanepoint_by_lane_memo[curr_lane.lane_id] = first_lanepoint
            shape_lanepoints.append(first_lanepoint)
            curr_lanepoint = first_lanepoint

            for p1, p2 in zip(lane_shape[1:], lane_shape[2:]):
                heading_ = vec_to_radians(p2 - p1)
                heading_ = Heading(heading_)
                orientation_ = fast_quaternion_from_angle(heading_)
                linked_lanepoint = LinkedLanePoint(
                    lp=LanePoint(
                        lane=self._road_map.lane_by_id(curr_lane.lane_id),
                        pose=Pose(position=p1, orientation=orientation_),
                    ),
                    nexts=[],
                    is_inferred=False,
                )

                shape_lanepoints.append(linked_lanepoint)
                curr_lanepoint.nexts.append(linked_lanepoint)
                curr_lanepoint = linked_lanepoint

            # Add a lanepoint for the last point of the current lane
            last_linked_lanepoint = LinkedLanePoint(
                lp=LanePoint(
                    lane=curr_lanepoint.lp.lane,
                    pose=Pose(
                        position=lane_shape[-1],
                        orientation=curr_lanepoint.lp.pose.orientation,
                    ),
                ),
                nexts=[],
                is_inferred=False,
            )

            shape_lanepoints.append(last_linked_lanepoint)
            curr_lanepoint.nexts.append(last_linked_lanepoint)
            curr_lanepoint = last_linked_lanepoint

            for out_lane in curr_lane.outgoing_lanes:
                if out_lane.is_drivable:
                    lane_queue.put((out_lane, curr_lanepoint))

        return initial_lanepoint, shape_lanepoints

    @staticmethod
    def _interpolate_shape_lanepoints(
        shape_lanepoints: Sequence[LinkedLanePoint], spacing: float
    ) -> List[LinkedLanePoint]:
        # memoize interpolated lanepoints on the shape lanepoint at start of
        # the line we are interpolating
        interp_memo = {}

        linked_lanepoints = []
        for shape_lp in shape_lanepoints:
            _, new_lanepoints = OpenDriveLanePoints._interpolate_from_shape_lp(
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
                is_inferred=False,
            )

            if previous_lp is not None:
                previous_lp.nexts.append(first_linked_lanepoint)

            if initial_lanepoint is None:
                initial_lanepoint = first_linked_lanepoint

            interp_memo[shape_lp.lp] = first_linked_lanepoint

            newly_created_lanepoints.append(first_linked_lanepoint)

            for current_shape_lp in shape_lp.nexts:
                next_shape_lp = OpenDriveLanePoints._process_interp_for_lane_lp(
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
        curr_lane = shape_lp.lp.lane

        curr_lanepoint = first_linked_lanepoint

        lane_seg_vec = np.subtract(
            next_shape_lp.lp.pose.position, shape_lp.lp.pose.position
        )
        lane_seg_len = np.linalg.norm(lane_seg_vec)

        # We set the initial distance into the lane at `spacing` because
        # we already have a lanepoint along this segment (curr_lanepoint)
        dist_into_lane_seg = spacing
        while dist_into_lane_seg < lane_seg_len:
            p = dist_into_lane_seg / lane_seg_len
            pos = shape_lp.lp.pose.position + lane_seg_vec * p

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
                0.5
                * np.subtract(
                    curr_lanepoint.lp.pose.position, next_shape_lp.lp.pose.position
                )
            )
            mid_point_current_next_shape_lp = 0.5 * (
                np.add(next_shape_lp.lp.pose.position, curr_lanepoint.lp.pose.position)
            )

            if half_distant_current_next_shape_lp < minimum_dist_next_shape_lp:
                pos = mid_point_current_next_shape_lp
            dist_pos_next_shape_lp = np.linalg.norm(
                np.subtract(next_shape_lp.lp.pose.position, pos)
            )
            if dist_pos_next_shape_lp < last_spacing_threshold_dist:
                break

            heading = vec_to_radians(lane_seg_vec)
            orientation = fast_quaternion_from_angle(heading)

            linked_lanepoint = LinkedLanePoint(
                lp=LanePoint(
                    lane=curr_lane,
                    pose=Pose(position=pos, orientation=orientation),
                ),
                nexts=[],
            )

            curr_lanepoint.nexts.append(linked_lanepoint)
            curr_lanepoint = linked_lanepoint
            newly_created_lanepoints.append(linked_lanepoint)
            dist_into_lane_seg += spacing
        return curr_lanepoint
