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
# FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# to allow for typing to refer to class being defined (LinkedLanePoint)
from __future__ import annotations

import queue
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import List, NamedTuple, Sequence, Tuple

import numpy as np
from scipy.spatial import KDTree

from smarts.core.coordinates import Heading, Point, Pose
from smarts.core.road_map import RoadMap
from smarts.core.utils.math import (
    fast_quaternion_from_angle,
    squared_dist,
    vec_to_radians,
)


@dataclass(frozen=True)
class LanePoint:
    """A point on a lane."""

    lane: RoadMap.Lane
    """The lane this point is on."""
    pose: Pose
    """The pose of this lane."""
    lane_width: float
    """The width of the lane at this point."""


class LinkedLanePoint(NamedTuple):
    """A lane point that is linked to the next points in the road network."""

    lp: LanePoint = None
    is_inferred: bool = True
    nexts: List[LinkedLanePoint] = []  # list of next immediate LanePoint(s)
    # it's a list of LanePoints because a path may branch at junctions

    def __hash__(self):
        ## distinguish between different continuations here too
        ## so the lru_cache on _lanepoint_paths_starting_at_lanepoint() below
        ## doesn't return the wrong set of LanePoints.
        return hash((self.lp, tuple(nlp.lp for nlp in self.nexts)))


class LanePoints:
    """A LanePoint utility class."""

    def __init__(self, shape_lps: List[LinkedLanePoint], spacing: float):
        # XXX: for a big map, may not want to cache ALL of the potential LanePoints
        #      nor waste time here finding all of them.
        #      Lanepoints might be generated on demand based upon edges and lookahead.
        self._linked_lanepoints = LanePoints._interpolate_shape_lanepoints(
            shape_lps, spacing
        )
        self._lanepoints_kd_tree = LanePoints._build_kd_tree(self._linked_lanepoints)

        self._lanepoints_by_lane_id = defaultdict(list)
        self._lanepoints_by_edge_id = defaultdict(list)
        for linked_lp in self._linked_lanepoints:
            lp_edge_id = linked_lp.lp.lane.road.road_id
            self._lanepoints_by_lane_id[linked_lp.lp.lane.lane_id].append(linked_lp)
            self._lanepoints_by_edge_id[lp_edge_id].append(linked_lp)

        self._lanepoints_kd_tree_by_lane_id = {
            lane_id: LanePoints._build_kd_tree(l_lps)
            for lane_id, l_lps in self._lanepoints_by_lane_id.items()
        }

        self._lanepoints_kd_tree_by_edge_id = {
            edge_id: LanePoints._build_kd_tree(l_lps)
            for edge_id, l_lps in self._lanepoints_by_edge_id.items()
        }

    @classmethod
    def from_sumo(
        cls,
        sumo_road_network,
        spacing,
    ):
        """Computes the lane shape (start/shape/end) lanepoints for all lanes in
        the network, the result of this function can be used to interpolate
        lanepoints along lanes to the desired granularity.
        """
        from smarts.core.utils.sumo import sumolib  # isort:skip
        from sumolib.net.edge import Edge  # isort:skip
        from sumolib.net.lane import Lane  # isort:skip
        from .sumo_road_network import SumoRoadNetwork

        assert type(sumo_road_network) == SumoRoadNetwork

        def _shape_lanepoints_along_lane(
            road_map: SumoRoadNetwork, lane: RoadMap.Lane, lanepoint_by_lane_memo: dict
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

                lane_shape = [np.array(p) for p in lane.getShape(False)]

                assert len(lane_shape) >= 2, repr(lane_shape)

                heading = vec_to_radians(lane_shape[1] - lane_shape[0])
                heading = Heading(heading)
                orientation = fast_quaternion_from_angle(heading)

                lane_width, _ = road_map.lane_by_id(lane.getID()).width_at_offset(0)
                first_lanepoint = LinkedLanePoint(
                    lp=LanePoint(
                        lane=road_map.lane_by_id(lane.getID()),
                        pose=Pose(position=lane_shape[0], orientation=orientation),
                        lane_width=lane_width,
                    ),
                    nexts=[],
                    is_inferred=False,
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
                    orientation_ = fast_quaternion_from_angle(heading_)
                    linked_lanepoint = LinkedLanePoint(
                        lp=LanePoint(
                            lane=road_map.lane_by_id(lane.getID()),
                            pose=Pose(position=p1, orientation=orientation_),
                            lane_width=lane_width,
                        ),
                        nexts=[],
                        is_inferred=False,
                    )

                    shape_lanepoints.append(linked_lanepoint)
                    curr_lanepoint.nexts.append(linked_lanepoint)
                    curr_lanepoint = linked_lanepoint

                # Add a lanepoint for the last point of the current lane
                lane_width, _ = curr_lanepoint.lp.lane.width_at_offset(0)
                last_linked_lanepoint = LinkedLanePoint(
                    lp=LanePoint(
                        lane=curr_lanepoint.lp.lane,
                        pose=Pose(
                            position=lane_shape[-1],
                            orientation=curr_lanepoint.lp.pose.orientation,
                        ),
                        lane_width=lane_width,
                    ),
                    nexts=[],
                    is_inferred=False,
                )

                shape_lanepoints.append(last_linked_lanepoint)
                curr_lanepoint.nexts.append(last_linked_lanepoint)
                curr_lanepoint = last_linked_lanepoint

                for out_connection in lane.getOutgoing():
                    out_lane = out_connection.getToLane()

                    # Use internal lanes of junctions (if we're at a junction)
                    via_lane_id = out_connection.getViaLaneID()
                    if via_lane_id:
                        out_lane = road_map._graph.getLane(via_lane_id)

                    lane_queue.put((out_lane, curr_lanepoint))

            return initial_lanepoint, shape_lanepoints

        # Don't request internal lanes since we get them by calling
        # `lane.getViaLaneID()`
        edges = sumo_road_network._graph.getEdges(False)
        lanepoint_by_lane_memo = {}
        shape_lps = []

        for edge in edges:
            for lane in edge.getLanes():
                _, new_lps = _shape_lanepoints_along_lane(
                    sumo_road_network, lane, lanepoint_by_lane_memo
                )
                shape_lps += new_lps

        return cls(shape_lps, spacing)

    @classmethod
    def from_opendrive(
        cls,
        od_road_network,
        spacing,
    ):
        """Computes the lane shape (start/shape/end) lanepoints for all lanes in
        the network, the result of this function can be used to interpolate
        lanepoints along lanes to the desired granularity.
        """
        from .opendrive_road_network import OpenDriveRoadNetwork

        assert type(od_road_network) == OpenDriveRoadNetwork

        def _shape_lanepoints_along_lane(
            lane: RoadMap.Lane,
            lanepoint_by_lane_memo: dict,
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

                lane_shape = [p.as_np_array[:2] for p in curr_lane.center_polyline]

                assert len(lane_shape) >= 2, repr(lane_shape)

                heading = vec_to_radians(lane_shape[1] - lane_shape[0])
                heading = Heading(heading)
                orientation = fast_quaternion_from_angle(heading)

                first_lane_coord = curr_lane.to_lane_coord(
                    Point(x=lane_shape[0][0], y=lane_shape[0][1], z=0.0)
                )

                lane_width, _ = curr_lane.width_at_offset(first_lane_coord.s)
                first_lanepoint = LinkedLanePoint(
                    lp=LanePoint(
                        lane=curr_lane,
                        pose=Pose(position=lane_shape[0], orientation=orientation),
                        lane_width=lane_width,
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
                    lp_lane_coord = curr_lane.to_lane_coord(
                        Point(x=p1[0], y=p1[1], z=0.0)
                    )
                    lane_width, _ = curr_lane.width_at_offset(lp_lane_coord.s)
                    linked_lanepoint = LinkedLanePoint(
                        lp=LanePoint(
                            lane=curr_lane,
                            pose=Pose(position=p1, orientation=orientation_),
                            lane_width=lane_width,
                        ),
                        nexts=[],
                        is_inferred=False,
                    )

                    shape_lanepoints.append(linked_lanepoint)
                    curr_lanepoint.nexts.append(linked_lanepoint)
                    curr_lanepoint = linked_lanepoint

                # Add a lanepoint for the last point of the current lane
                last_lane_coord = curr_lanepoint.lp.lane.to_lane_coord(
                    Point(x=lane_shape[-1][0], y=lane_shape[-1][1], z=0.0)
                )
                lane_width, _ = curr_lanepoint.lp.lane.width_at_offset(
                    last_lane_coord.s
                )
                last_linked_lanepoint = LinkedLanePoint(
                    lp=LanePoint(
                        lane=curr_lanepoint.lp.lane,
                        pose=Pose(
                            position=lane_shape[-1],
                            orientation=curr_lanepoint.lp.pose.orientation,
                        ),
                        lane_width=lane_width,
                    ),
                    nexts=[],
                    is_inferred=False,
                )

                shape_lanepoints.append(last_linked_lanepoint)
                curr_lanepoint.nexts.append(last_linked_lanepoint)
                curr_lanepoint = last_linked_lanepoint

                outgoing_roads_added = []
                for out_lane in curr_lane.outgoing_lanes:
                    if out_lane.is_drivable:
                        lane_queue.put((out_lane, curr_lanepoint))
                    outgoing_road = out_lane.road
                    if out_lane.road not in outgoing_roads_added:
                        outgoing_roads_added.append(outgoing_road)
                for outgoing_road in outgoing_roads_added:
                    for next_lane in outgoing_road.lanes:
                        if (
                            next_lane.is_drivable
                            and next_lane not in curr_lane.outgoing_lanes
                        ):
                            lane_queue.put((next_lane, curr_lanepoint))

            return initial_lanepoint, shape_lanepoints

        roads = od_road_network._roads
        lanepoint_by_lane_memo = {}
        shape_lps = []

        for road_id in roads:
            road = roads[road_id]
            # go ahead and add lanepoints for composite lanes,
            # even though we don't on other map formats,
            # and then filter these out on lanepoint queries.
            # (not an issue right now for OpenDrive since we don't
            # find composite lanes, but it may be in the future.)
            for lane in road.lanes:
                # Ignore non drivable lanes in OpenDRIVE
                if lane.is_drivable:
                    _, new_lps = _shape_lanepoints_along_lane(
                        lane, lanepoint_by_lane_memo
                    )
                    shape_lps += new_lps

        return cls(shape_lps, spacing)

    @classmethod
    def from_waymo(
        cls,
        waymo_road_network,
        spacing,
    ):
        """Computes the lane shape (start/shape/end) lanepoints for all lanes in
        the network, the result of this function can be used to interpolate
        lanepoints along lanes to the desired granularity.
        """
        from .waymo_map import WaymoMap

        assert type(waymo_road_network) == WaymoMap

        def _shape_lanepoints_along_lane(
            lane: RoadMap.Lane,
            lanepoint_by_lane_memo: dict,
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

                lane_shape = curr_lane._lane_pts

                assert (
                    len(lane_shape) >= 2
                ), f"{repr(lane_shape)} for lane_id={curr_lane.lane_id}"

                vd = lane_shape[1] - lane_shape[0]
                heading = Heading(vec_to_radians(vd[:2]))
                orientation = fast_quaternion_from_angle(heading)

                lane_width, _ = curr_lane.width_at_offset(0)
                first_lanepoint = LinkedLanePoint(
                    lp=LanePoint(
                        lane=curr_lane,
                        pose=Pose(position=lane_shape[0], orientation=orientation),
                        lane_width=lane_width,
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
                    vd = p2 - p1
                    heading_ = Heading(vec_to_radians(vd[:2]))
                    orientation_ = fast_quaternion_from_angle(heading_)
                    lane_width, _ = curr_lane.width_at_offset(0)
                    linked_lanepoint = LinkedLanePoint(
                        lp=LanePoint(
                            lane=curr_lane,
                            pose=Pose(position=p1, orientation=orientation_),
                            lane_width=lane_width,
                        ),
                        nexts=[],
                        is_inferred=False,
                    )

                    shape_lanepoints.append(linked_lanepoint)
                    curr_lanepoint.nexts.append(linked_lanepoint)
                    curr_lanepoint = linked_lanepoint

                # Add a lanepoint for the last point of the current lane
                curr_lanepoint_lane = curr_lanepoint.lp.lane
                lane_width, _ = curr_lanepoint_lane.width_at_offset(0)
                last_linked_lanepoint = LinkedLanePoint(
                    lp=LanePoint(
                        lane=curr_lanepoint.lp.lane,
                        pose=Pose(
                            position=lane_shape[-1],
                            orientation=curr_lanepoint.lp.pose.orientation,
                        ),
                        lane_width=lane_width,
                    ),
                    nexts=[],
                    is_inferred=False,
                )

                shape_lanepoints.append(last_linked_lanepoint)
                curr_lanepoint.nexts.append(last_linked_lanepoint)
                curr_lanepoint = last_linked_lanepoint

                for out_lane in curr_lane.outgoing_lanes:
                    if out_lane and out_lane.is_drivable:
                        lane_queue.put((out_lane, curr_lanepoint))
            return initial_lanepoint, shape_lanepoints

        roads = waymo_road_network._roads
        lanepoint_by_lane_memo = {}
        shape_lps = []

        for road_id in roads:
            road = roads[road_id]
            # go ahead and add lanepoints for composite lanes,
            # even though we don't on other map formats,
            # and then filter these out on lanepoint queries.
            for lane in road.lanes:
                # Ignore non drivable lanes in Waymo
                if lane.is_drivable:
                    _, new_lps = _shape_lanepoints_along_lane(
                        lane, lanepoint_by_lane_memo
                    )
                    shape_lps += new_lps

        return cls(shape_lps, spacing)

    @staticmethod
    def _build_kd_tree(linked_lps: Sequence[LinkedLanePoint]) -> KDTree:
        return KDTree(
            np.array([l_lp.lp.pose.point.as_np_array[:2] for l_lp in linked_lps]),
            leafsize=50,
        )

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
                is_inferred=False,
            )

            if previous_lp is not None:
                previous_lp.nexts.append(first_linked_lanepoint)

            if initial_lanepoint is None:
                initial_lanepoint = first_linked_lanepoint

            interp_memo[shape_lp.lp] = first_linked_lanepoint

            newly_created_lanepoints.append(first_linked_lanepoint)

            for current_shape_lp in shape_lp.nexts:
                if (
                    current_shape_lp.lp.lane.lane_id == shape_lp.lp.lane.lane_id
                    or current_shape_lp.lp.lane in shape_lp.lp.lane.outgoing_lanes
                ):
                    next_shape_lp = LanePoints._process_interp_for_lane_lp(
                        shape_lp,
                        first_linked_lanepoint,
                        current_shape_lp,
                        spacing,
                        newly_created_lanepoints,
                    )
                    shape_queue.put((current_shape_lp, next_shape_lp))
                else:
                    shape_queue.put((current_shape_lp, first_linked_lanepoint))
        return initial_lanepoint, newly_created_lanepoints

    @staticmethod
    def _process_interp_for_lane_lp(
        shape_lp: LinkedLanePoint,
        first_linked_lanepoint: LinkedLanePoint,
        next_shape_lp: LinkedLanePoint,
        spacing: float,
        newly_created_lanepoints: List[LinkedLanePoint],
    ) -> LinkedLanePoint:
        rmlane = shape_lp.lp.lane

        curr_lanepoint = first_linked_lanepoint

        lane_seg_vec = (
            next_shape_lp.lp.pose.as_position2d() - shape_lp.lp.pose.as_position2d()
        )
        lane_seg_len = np.linalg.norm(lane_seg_vec)

        # We set the initial distance into the lane at `spacing` because
        # we already have a lanepoint along this segment (curr_lanepoint)
        dist_into_lane_seg = spacing
        while dist_into_lane_seg < lane_seg_len:
            p = dist_into_lane_seg / lane_seg_len
            pos = shape_lp.lp.pose.as_position2d() + lane_seg_vec * p

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
                * (
                    curr_lanepoint.lp.pose.as_position2d()
                    - next_shape_lp.lp.pose.as_position2d()
                )
            )
            mid_point_current_next_shape_lp = 0.5 * (
                next_shape_lp.lp.pose.as_position2d()
                + curr_lanepoint.lp.pose.as_position2d()
            )
            if half_distant_current_next_shape_lp < minimum_dist_next_shape_lp:
                pos = mid_point_current_next_shape_lp
            dist_pos_next_shape_lp = np.linalg.norm(
                next_shape_lp.lp.pose.as_position2d() - pos
            )
            if dist_pos_next_shape_lp < last_spacing_threshold_dist:
                break

            heading = vec_to_radians(lane_seg_vec)
            orientation = fast_quaternion_from_angle(heading)

            rmlane_coord = rmlane.to_lane_coord(Point(x=pos[0], y=pos[1], z=0.0))
            lane_width, _ = rmlane.width_at_offset(rmlane_coord.s)
            linked_lanepoint = LinkedLanePoint(
                lp=LanePoint(
                    lane=rmlane,
                    pose=Pose(position=pos, orientation=orientation),
                    lane_width=lane_width,
                ),
                nexts=[],
                is_inferred=True,
            )

            curr_lanepoint.nexts.append(linked_lanepoint)
            curr_lanepoint = linked_lanepoint
            newly_created_lanepoints.append(linked_lanepoint)
            dist_into_lane_seg += spacing
        return curr_lanepoint

    @staticmethod
    def _closest_linked_lp_in_kd_tree_batched(
        points: Sequence[Point],
        linked_lps,
        tree: KDTree,
        k: int = 1,
        filter_composites: bool = False,
    ):
        p2ds = np.array([p.as_np_array[:2] for p in points])
        _, closest_indices = tree.query(p2ds, k=min(k, len(linked_lps)))
        closest_indices = np.atleast_2d(closest_indices)
        if filter_composites:
            result = [
                [
                    linked_lps[idx]
                    for idx in idxs
                    if not linked_lps[idx].lp.lane.is_composite
                ]
                for idxs in closest_indices
            ]
            if result:
                return result
        # if filtering, only return lanepoints in composite lanes if we didn't hit any in simple lanes...
        return [[linked_lps[idx] for idx in idxs] for idxs in closest_indices]

    @staticmethod
    def _closest_linked_lp_in_kd_tree_with_pose_batched(
        poses,
        lanepoints,
        tree,
        within_radius: float,
        k: int = 10,
        filter_composites: bool = False,
    ):
        linked_lanepoints = LanePoints._closest_linked_lp_in_kd_tree_batched(
            [pose.point for pose in poses],
            lanepoints,
            tree,
            k=k,
            filter_composites=filter_composites,
        )

        linked_lanepoints = [
            sorted(
                l_lps,
                key=lambda _llp: squared_dist(
                    poses[idx].as_position2d(), _llp.lp.pose.as_position2d()
                ),
            )
            for idx, l_lps in enumerate(linked_lanepoints)
        ]
        # exclude those outside radius except closest
        if within_radius is not None:
            radius_sq = within_radius * within_radius
            linked_lanepoints = [
                [
                    _llp
                    for i, _llp in enumerate(_llps)
                    if squared_dist(
                        poses[idx].as_position2d(), _llp.lp.pose.as_position2d()
                    )
                    <= radius_sq
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
                [pose.point for _, pose in unfound_lanepoints],
                lanepoints,
                tree=tree,
                k=k,
                filter_composites=filter_composites,
            )
            # Replace the empty lanepoint locations
            for (i, _), lps in [
                g for g in zip(unfound_lanepoints, remaining_linked_lps)
            ]:
                linked_lanepoints[i] = [lps]

        return [
            sorted(
                l_lps,
                key=lambda _llp: squared_dist(
                    poses[idx].as_position2d(), _llp.lp.pose.as_position2d()
                )
                + abs(poses[idx].heading.relative_to(_llp.lp.pose.heading)),
            )
            for idx, l_lps in enumerate(linked_lanepoints)
        ]

    def closest_lanepoints(
        self,
        poses: Sequence[Pose],
        within_radius: float = 10,
        maximum_count: int = 10,
    ) -> List[LanePoint]:
        """Get the lanepoints closest to the given poses.
        Args:
            poses:
                The poses to look around for lanepoints.
            within_radius:
                The radius which lanepoints can be found from the given poses.
            maximum_count:
                The maximum lanepoints found.
        """
        lanepoints = self._linked_lanepoints
        kd_tree = self._lanepoints_kd_tree
        linked_lanepoints = LanePoints._closest_linked_lp_in_kd_tree_with_pose_batched(
            poses,
            lanepoints,
            kd_tree,
            within_radius=within_radius,
            k=maximum_count,
            filter_composites=True,
        )
        return [l_lps[0].lp for l_lps in linked_lanepoints]

    def closest_lanepoint_on_lane_to_point(self, point, lane_id: str) -> LanePoint:
        """Returns the closest lanepoint on the given lane to the given world coordinate."""
        return self.closest_linked_lanepoint_on_lane_to_point(point, lane_id).lp

    def closest_linked_lanepoint_on_lane_to_point(
        self, point: Point, lane_id: str
    ) -> LinkedLanePoint:
        """Returns the closest linked lanepoint on the given lane."""
        lane_kd_tree = self._lanepoints_kd_tree_by_lane_id[lane_id]
        return LanePoints._closest_linked_lp_in_kd_tree_batched(
            [point], self._lanepoints_by_lane_id[lane_id], lane_kd_tree, k=1
        )[0][0]

    def closest_linked_lanepoint_on_road(
        self, point: Point, road_id: str
    ) -> LinkedLanePoint:
        """Returns the closest linked lanepoint on the given road."""
        return LanePoints._closest_linked_lp_in_kd_tree_batched(
            [point],
            self._lanepoints_by_edge_id[road_id],
            self._lanepoints_kd_tree_by_edge_id[road_id],
        )[0][0]

    @lru_cache(maxsize=32)
    def paths_starting_at_lanepoint(
        self, lanepoint: LinkedLanePoint, lookahead: int, filter_edge_ids: tuple
    ) -> List[List[LinkedLanePoint]]:
        """Returns all full branches from the given lanepoint up to the length of the lookahead.
        Args:
            lanepoint:
                The starting lanepoint.
            lookahead:
                The maximum lanepoints in a branch.
            filter_edge_ids:
                Whitelisted edge ids.
        Returns:
            All branches(as lists) stemming from the lanepoint.
        """
        lanepoint_paths = [[lanepoint]]
        for _ in range(lookahead):
            next_lanepoint_paths = []
            for path in lanepoint_paths:
                branching_paths = []
                for next_lp in path[-1].nexts:
                    # TODO: This could be a problem for SUMO. What about internal lanes?
                    # Filter only the edges we're interested in
                    next_lane = next_lp.lp.lane
                    edge_id = next_lane.road.road_id
                    if filter_edge_ids and edge_id not in filter_edge_ids:
                        continue
                    if (
                        filter_edge_ids
                        and edge_id != filter_edge_ids[-1]
                        and all(
                            out_lane.road.road_id not in filter_edge_ids
                            for out_lane in next_lane.outgoing_lanes
                        )
                    ):
                        continue
                    new_path = path + [next_lp]
                    branching_paths.append(new_path)

                if not branching_paths:
                    branching_paths = [path]

                next_lanepoint_paths += branching_paths

            lanepoint_paths = next_lanepoint_paths

        return lanepoint_paths
