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
from typing import TYPE_CHECKING, List, NamedTuple, Sequence, Tuple

import numpy as np

from smarts.core.coordinates import Heading, Point, Pose
from smarts.core.utils.core_math import fast_quaternion_from_angle, vec_to_radians

if TYPE_CHECKING:
    from smarts.core.road_map import RoadMap


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
        #      Lanepoints might be generated on demand based upon edges and look-ahead.
        self._linked_lanepoints = LanePoints._interpolate_shape_lanepoints(
            shape_lps, spacing
        )
        self._lp_points = np.array(
            [l_lp.lp.pose.point.as_np_array[:2] for l_lp in self._linked_lanepoints]
        )

        self._lanepoints_by_lane_id = defaultdict(list)
        self._lanepoints_by_edge_id = defaultdict(list)
        for linked_lp in self._linked_lanepoints:
            lp_edge_id = linked_lp.lp.lane.road.road_id
            self._lanepoints_by_lane_id[linked_lp.lp.lane.lane_id].append(linked_lp)
            self._lanepoints_by_edge_id[lp_edge_id].append(linked_lp)

        self._lp_points_by_lane_id = {
            lane_id: np.array([l_lp.lp.pose.point.as_np_array[:2] for l_lp in l_lps])
            for lane_id, l_lps in self._lanepoints_by_lane_id.items()
        }

        self._lp_points_by_edge_id = {
            edge_id: np.array([l_lp.lp.pose.point.as_np_array[:2] for l_lp in l_lps])
            for edge_id, l_lps in self._lanepoints_by_edge_id.items()
        }

    @classmethod
    def from_sumo(
        cls,
        sumo_road_network,
        spacing,
    ):
        """Computes the lane shape (start/shape/end) lane-points for all lanes in
        the network, the result of this function can be used to interpolate
        lane-points along lanes to the desired granularity.
        """
        from smarts.core.utils.sumo_utils import sumolib  # isort:skip
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

                # Add a lane-point for the last point of the current lane
                lane_width, _ = curr_lanepoint.lp.lane.width_at_offset(0)
                last_linked_lanepoint = LinkedLanePoint(
                    lp=LanePoint(
                        lane=curr_lanepoint.lp.lane,
                        pose=Pose(
                            position=lane_shape[-1][:2],
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
        """Computes the lane shape (start/shape/end) lane-points for all lanes in
        the network, the result of this function can be used to interpolate
        lane-points along lanes to the desired granularity.
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

                # Add a lane-point for the last point of the current lane
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
            # go ahead and add lane-points for composite lanes,
            # even though we don't on other map formats,
            # and then filter these out on lane-point queries.
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
        """Computes the lane shape (start/shape/end) lane-points for all lanes in
        the network, the result of this function can be used to interpolate
        lane-points along lanes to the desired granularity.
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
            # go ahead and add lane-points for composite lanes,
            # even though we don't on other map formats,
            # and then filter these out on lane-point queries.
            for lane in road.lanes:
                # Ignore non drivable lanes in Waymo
                if lane.is_drivable:
                    _, new_lps = _shape_lanepoints_along_lane(
                        lane, lanepoint_by_lane_memo
                    )
                    shape_lps += new_lps

        return cls(shape_lps, spacing)

    @classmethod
    def from_argoverse(
        cls,
        argoverse_map,
        spacing,
    ):
        """Computes the lane shape (start/shape/end) lane-points for all lanes in
        the network, the result of this function can be used to interpolate
        lane-points along lanes to the desired granularity.
        """
        from .argoverse_map import ArgoverseMap

        assert type(argoverse_map) == ArgoverseMap

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

                lane_shape = curr_lane._centerline

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

        roads = argoverse_map._roads
        lanepoint_by_lane_memo = {}
        shape_lps = []

        for road in roads.values():
            for lane in road.lanes:
                if lane.is_drivable:
                    _, new_lps = _shape_lanepoints_along_lane(
                        lane, lanepoint_by_lane_memo
                    )
                    shape_lps += new_lps

        return cls(shape_lps, spacing)

    @staticmethod
    def _interpolate_shape_lanepoints(
        shape_lanepoints: Sequence[LinkedLanePoint], spacing: float
    ) -> List[LinkedLanePoint]:
        # memoize interpolated lane-points on the shape lane-point at start of
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
                lp=shape_lp.lp,  # lane-points are frozen, so no need to copy lp here
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
        # we already have a lane-point along this segment (curr_lanepoint)
        dist_into_lane_seg = spacing
        while dist_into_lane_seg < lane_seg_len:
            p = dist_into_lane_seg / lane_seg_len
            pos = shape_lp.lp.pose.as_position2d() + lane_seg_vec * p

            # The thresholds for calculating last lane-point. If the
            # midpoint between the current lane-point and the next shape
            # lane-point is less than the minimum distance then the last
            # lane-point position will be that midpoint. If the midpoint
            # is closer than last spacing threshold to the next shape
            # lane-point, then the last lane-point will be the current
            # lane-point.
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
    def _closest_linked_lp_to_point(
        point: Point,
        linked_lps: List[LinkedLanePoint],
        points: np.ndarray,
        k: int = 1,
        filter_composites: bool = False,
    ) -> List[LinkedLanePoint]:
        x = point.as_np_array[:2]
        dists = np.sqrt(np.sum((points - x) ** 2, axis=1))
        closest_indices = np.argsort(dists)[:k]

        if filter_composites:
            result = [
                linked_lps[idx]
                for idx in closest_indices
                if not linked_lps[idx].lp.lane.is_composite
            ]
            if result:
                return result

        # if filtering, only return lane-points in composite lanes if we didn't hit any in simple lanes...
        return [linked_lps[idx] for idx in closest_indices]

    def closest_lanepoints(
        self,
        pose: Pose,
        maximum_count: int = 10,
    ) -> List[LanePoint]:
        """Get the lane-points closest to the given pose.
        Args:
            pose:
                The pose to look around for lane-points.
            maximum_count:
                The maximum number of lane-points that should be found.
        """
        linked_lanepoints = LanePoints._closest_linked_lp_to_point(
            pose.point,
            self._linked_lanepoints,
            self._lp_points,
            k=maximum_count,
            filter_composites=True,
        )
        return [llp.lp for llp in linked_lanepoints]

    def closest_linked_lanepoint_on_lane_to_point(
        self, point: Point, lane_id: str
    ) -> LinkedLanePoint:
        """Returns the closest linked lane-point on the given lane."""
        return LanePoints._closest_linked_lp_to_point(
            point,
            self._lanepoints_by_lane_id[lane_id],
            self._lp_points_by_lane_id[lane_id],
            k=1,
        )[0]

    def closest_linked_lanepoint_on_road(
        self, point: Point, road_id: str
    ) -> LinkedLanePoint:
        """Returns the closest linked lane-point on the given road."""
        return LanePoints._closest_linked_lp_to_point(
            point,
            self._lanepoints_by_edge_id[road_id],
            self._lp_points_by_edge_id[road_id],
        )[0]

    @lru_cache(maxsize=32)
    def paths_starting_at_lanepoint(
        self, lanepoint: LinkedLanePoint, lookahead: int, route_edge_ids: tuple
    ) -> List[List[LinkedLanePoint]]:
        """Returns all full branches from the given lane-point up to the length of the look-ahead.
        Branches will be filtered at the lane level if they or their outgoing lanes do not belong
        to a road in the route edge list.
        Args:
            lanepoint (LinkedLanePoint):
                The starting lane-point.
            lookahead (int):
                The maximum lane-points in a branch.
            route_edge_ids (Tuple[str]):
                White-listed edge ids for a route.
        Returns:
            All branches (as lists) stemming from the input lane-point.
        """
        # Early exit if there are no valid paths ahead in the route for this lane
        lp_lane = lanepoint.lp.lane
        if (
            route_edge_ids
            and lp_lane.road.road_id != route_edge_ids[-1]
            and all(
                out_lane.road.road_id not in route_edge_ids
                for out_lane in lp_lane.outgoing_lanes
            )
        ):
            return []

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
                    if route_edge_ids and edge_id not in route_edge_ids:
                        continue
                    if (
                        route_edge_ids
                        and edge_id != route_edge_ids[-1]
                        and all(
                            out_lane.road.road_id not in route_edge_ids
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
