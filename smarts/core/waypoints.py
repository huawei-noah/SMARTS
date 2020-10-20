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
import queue
import random
from typing import Sequence
from dataclasses import dataclass
from collections import namedtuple, defaultdict

import numpy as np
from sklearn.neighbors import KDTree

from .coordinates import Heading, Pose
from .utils.math import (
    vec_to_radians,
    lerp,
    vec_2d,
    signed_dist_to_line,
)


@dataclass(frozen=True)
class Waypoint:
    id: int  # Numeric identifier for this waypoint
    pos: np.ndarray  # Center point of lane
    heading: Heading  # Heading angle of lane at this point (radians)
    lane_width: float  # Width of lane at this point (meters)
    speed_limit: float  # Lane speed in m/s
    lane_id: str  # ID of lane under waypoint
    right_of_way: bool  # True if this waypoint has right of way, False otherwise
    lane_index: int  # Index of the lane this waypoint is over. 0 is the outer(right) most lane.

    def dist_to(self, p):
        """Calculates straight line distance to the given 2D point"""
        return np.linalg.norm(self.pos - vec_2d(p))

    def relative_heading(self, h: Heading):
        """Computes relative heading between the given angle and the waypoint heading

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
        line formed by the waypoint position and the waypoints heading.

        Negative signals right of line and Positive left of line
        """
        return signed_dist_to_line(p, self.pos, self.heading.direction_vector())

    @property
    def pose(self):
        return Pose.from_center(tuple(self.pos), self.heading)


LinkedWaypoint = namedtuple(
    "LinkedWaypoint",
    [
        "wp",  # Waypoint: current waypoint
        "nexts",  # list of LinkedWaypoint: list of next immediate waypoints
        # it's a list of waypoints because a path may branch at junctions
    ],
)


class Waypoints:
    def __init__(self, road_network, spacing, debug=True):
        self.spacing = spacing
        self._next_waypoint_id = 0
        self._road_network = road_network

        shape_wps = self._shape_waypoints()
        self._linked_waypoints = self._interpolate_shape_waypoints(shape_wps, spacing)

        self._waypoints_kd_tree = self._build_kd_tree(self._linked_waypoints)

        self._waypoints_by_lane_id = defaultdict(list)
        self._waypoints_by_edge_id = defaultdict(list)
        for linked_wp in self._linked_waypoints:
            wp_edge_id = self._edge(linked_wp.wp).getID()
            self._waypoints_by_lane_id[linked_wp.wp.lane_id].append(linked_wp)
            self._waypoints_by_edge_id[wp_edge_id].append(linked_wp)

        self._waypoints_kd_tree_by_lane_id = {
            lane_id: self._build_kd_tree(l_wps)
            for lane_id, l_wps in self._waypoints_by_lane_id.items()
        }

        self._waypoints_kd_tree_by_edge_id = {
            edge_id: self._build_kd_tree(l_wps)
            for edge_id, l_wps in self._waypoints_by_edge_id.items()
        }

    def _build_kd_tree(self, linked_wps):
        return KDTree(np.array([l_wp.wp.pos for l_wp in linked_wps]), leaf_size=50)

    def random_waypoint(self):
        return random.choice(self._linked_waypoints).wp

    def random_waypoints(self, n, replace=False):
        rselect = random.choices if replace else random.sample
        return [linked_wp.wp for linked_wp in rselect(self._linked_waypoints, k=n)]

    def closest_waypoint_batched(self, points):
        linked_waypoints = self._closest_linked_wp_in_kd_tree_batched(
            points, self._linked_waypoints, self._waypoints_kd_tree
        )

        return [l_wps[0].wp for l_wps in linked_waypoints]

    def closest_waypoint(self, point):
        linked_waypoint = self._closest_linked_wp_in_kd_tree_batched(
            [point], self._linked_waypoints, self._waypoints_kd_tree
        )[0][0]

        return linked_waypoint.wp

    def closest_waypoints(self, point, desired_count):
        linked_waypoints = self._closest_linked_wp_in_kd_tree_batched(
            [point], self._linked_waypoints, self._waypoints_kd_tree, k=desired_count
        )[0]

        return [lwp.wp for lwp in linked_waypoints]

    def closest_waypoint_on_lane(self, point, lane_id):
        lane_kd_tree = self._waypoints_kd_tree_by_lane_id[lane_id]
        linked_waypoint = self._closest_linked_wp_in_kd_tree_batched(
            [point], self._waypoints_by_lane_id[lane_id], lane_kd_tree
        )[0][0]

        return linked_waypoint.wp

    def waypoint_paths_on_lane_at(
        self, point, lane_id, lookahead, filter_edge_ids: Sequence[str] = None
    ):
        lane_kd_tree = self._waypoints_kd_tree_by_lane_id[lane_id]
        closest_linked_wp = self._closest_linked_wp_in_kd_tree_batched(
            [point], self._waypoints_by_lane_id[lane_id], lane_kd_tree, k=1
        )[0][0]

        waypoint_paths = self._waypoints_starting_at_waypoint(
            closest_linked_wp, lookahead, filter_edge_ids
        )

        # don't give users access to the linked structure
        unlinked_waypoint_paths = [
            [linked_wp.wp for linked_wp in path] for path in waypoint_paths
        ]
        return unlinked_waypoint_paths

    def waypoint_paths_at(self, point, lookahead):
        closest_linked_wp = self._closest_linked_wp_in_kd_tree_batched(
            [point], self._linked_waypoints, self._waypoints_kd_tree
        )[0][0]
        closest_lane = self._road_network.lane_by_id(closest_linked_wp.wp.lane_id)

        waypoint_paths = []
        for lane in closest_lane.getEdge().getLanes():
            lane_id = lane.getID()
            waypoint_paths += self.waypoint_paths_on_lane_at(point, lane_id, lookahead)

        sorted_wps = sorted(waypoint_paths, key=lambda p: p[0].lane_index)
        return sorted_wps

    def waypoint_paths_along_route(self, point, lookahead, route):
        assert len(route) > 0, f"Expected at least 1 edge in the route, got: {route}"
        closest_wp_on_each_route_edge = [
            self._closest_linked_wp_in_kd_tree_batched(
                [point],
                self._waypoints_by_edge_id[edge],
                self._waypoints_kd_tree_by_edge_id[edge],
            )[0][0]
            for edge in route
        ]

        closest_linked_wp = min(
            closest_wp_on_each_route_edge, key=lambda l_wp: l_wp.wp.dist_to(point)
        )
        closest_lane = self._road_network.lane_by_id(closest_linked_wp.wp.lane_id)

        waypoint_paths = []
        for lane in closest_lane.getEdge().getLanes():
            lane_id = lane.getID()
            waypoint_paths += self.waypoint_paths_on_lane_at(
                point, lane_id, lookahead, route
            )

        sorted_wps = sorted(waypoint_paths, key=lambda p: p[0].lane_index)
        return sorted_wps

    def _does_lane_have_right_of_way(self, lane):
        # This is hard coded for the competition scenario, we should use something like
        #  - prohibitions at the nearest intersection
        #  - Node.areFoes(lane1_index, lane2_index)
        return lane.getEdge().getPriority() > 1

    def _unique_waypoint_id(self):
        id_ = self._next_waypoint_id
        self._next_waypoint_id += 1
        return id_

    def _closest_linked_wp_in_kd_tree_batched(self, points, linked_wps, tree, k=1):
        p2ds = np.array([vec_2d(p) for p in points])
        closest_indices = tree.query(
            p2ds, k=k, return_distance=False, sort_results=True
        )

        return [[linked_wps[idx] for idx in idxs] for idxs in closest_indices]

    def _waypoints_starting_at_waypoint(
        self, waypoint: LinkedWaypoint, lookahead, filter_edge_ids: Sequence[str] = None
    ):
        waypoint_paths = [[waypoint]]
        for _ in range(lookahead):
            next_waypoint_paths = []
            for path in waypoint_paths:
                branching_paths = []
                for next_wp in path[-1].nexts:
                    # TODO: This could be a problem. What about internal lanes?
                    # Filter only the edges we're interested in
                    edge_id = self._edge(next_wp.wp).getID()
                    if filter_edge_ids and edge_id not in filter_edge_ids:
                        continue

                    new_path = path + [next_wp]
                    branching_paths.append(new_path)

                if branching_paths == []:
                    branching_paths = [path]

                next_waypoint_paths += branching_paths

            waypoint_paths = next_waypoint_paths

        return waypoint_paths

    def _interpolate_shape_waypoints(self, shape_waypoints, spacing):
        # memoize interpolated waypoints on the shape waypoint at start of
        # the line we are interpolating
        interp_memo = {}

        linked_waypoints = []
        for shape_wp in shape_waypoints:
            _, new_waypoints = self._interpolate_from_shape_wp(
                shape_wp, spacing, interp_memo
            )
            linked_waypoints += new_waypoints

        return linked_waypoints

    def _interpolate_from_shape_wp(self, shape_wp, spacing, interp_memo):
        shape_queue = queue.Queue()
        shape_queue.put((shape_wp, None))
        newly_created_waypoints = []
        initial_waypoint = None

        while not shape_queue.empty():
            shape_wp, previous_wp = shape_queue.get()
            if shape_wp.wp.id in interp_memo:
                if previous_wp is None:
                    continue
                previous_wp.nexts.append(interp_memo[shape_wp.wp.id])
                continue

            first_linked_waypoint = LinkedWaypoint(
                wp=Waypoint(
                    id=self._unique_waypoint_id(),
                    pos=shape_wp.wp.pos,
                    heading=shape_wp.wp.heading,
                    lane_width=shape_wp.wp.lane_width,
                    speed_limit=shape_wp.wp.speed_limit,
                    lane_id=shape_wp.wp.lane_id,
                    lane_index=shape_wp.wp.lane_index,
                    right_of_way=shape_wp.wp.right_of_way,
                ),
                nexts=[],
            )

            if previous_wp is not None:
                previous_wp.nexts.append(first_linked_waypoint)

            if initial_waypoint is None:
                initial_waypoint = first_linked_waypoint

            interp_memo[shape_wp.wp.id] = first_linked_waypoint

            newly_created_waypoints.append(first_linked_waypoint)

            for current_shape_wp in shape_wp.nexts:
                next_shape_wp = self._process_interp_for_lane_wp(
                    shape_wp,
                    first_linked_waypoint,
                    current_shape_wp,
                    spacing,
                    newly_created_waypoints,
                )

                shape_queue.put((current_shape_wp, next_shape_wp))

        return initial_waypoint, newly_created_waypoints

    def _process_interp_for_lane_wp(
        self,
        shape_wp,
        first_linked_waypoint,
        next_shape_wp,
        spacing,
        newly_created_waypoints,
    ):
        lane_id = shape_wp.wp.lane_id

        curr_waypoint = first_linked_waypoint

        lane_seg_vec = next_shape_wp.wp.pos - shape_wp.wp.pos
        lane_seg_len = np.linalg.norm(lane_seg_vec)

        # We set the initial distance into the lane at `spacing` because
        # we already have a waypoint along this segment (curr_waypoint)
        dist_into_lane_seg = spacing
        while dist_into_lane_seg < lane_seg_len:
            p = dist_into_lane_seg / lane_seg_len
            pos = shape_wp.wp.pos + lane_seg_vec * p

            # The thresholds for calculating last waypoint. If the
            # midpoint between the current waypoint and the next shap
            # waypoint is less than the minimum distant then the last
            # waypoint position will be that midpoint. If the midpoint
            # is closer than last spacing threshold to the next shape
            # waypoint, then the last waypoint will be the current
            # waypoint.
            last_spacing_threshold_dist = 0.8 * spacing
            minimum_dist_next_shape_wp = 1.4

            half_distant_current_next_shape_wp = np.linalg.norm(
                0.5 * (curr_waypoint.wp.pos - next_shape_wp.wp.pos)
            )
            mid_point_current_next_shape_wp = 0.5 * (
                next_shape_wp.wp.pos + curr_waypoint.wp.pos
            )
            if half_distant_current_next_shape_wp < minimum_dist_next_shape_wp:
                pos = mid_point_current_next_shape_wp
            dist_pos_next_shape_wp = np.linalg.norm(next_shape_wp.wp.pos - pos)
            if dist_pos_next_shape_wp < last_spacing_threshold_dist:
                break

            heading = vec_to_radians(lane_seg_vec)
            lane_width = lerp(shape_wp.wp.lane_width, next_shape_wp.wp.lane_width, p)
            speed_limit = lerp(shape_wp.wp.speed_limit, next_shape_wp.wp.speed_limit, p)

            # We don't consider the next_shape_wp.wp.right_of_way because you have the
            # right of way if the lane you started on has the right of way
            right_of_way = shape_wp.wp.right_of_way

            linked_waypoint = LinkedWaypoint(
                wp=Waypoint(
                    id=self._unique_waypoint_id(),
                    pos=pos,
                    heading=Heading(heading),
                    lane_width=lane_width,
                    speed_limit=speed_limit,
                    lane_id=lane_id,
                    lane_index=shape_wp.wp.lane_index,
                    right_of_way=right_of_way,
                ),
                nexts=[],
            )

            curr_waypoint.nexts.append(linked_waypoint)
            curr_waypoint = linked_waypoint
            newly_created_waypoints.append(linked_waypoint)
            dist_into_lane_seg += spacing
        return curr_waypoint

    def _shape_waypoints(self):
        """Computes the lane shape (start/shape/end) waypoints for all lanes in
        the network, the result of this function can be used to interpolate
        waypoints along lanes to the desired granularity.
        """
        # Don't request internal lanes since we get them by calling
        # `lane.getViaLaneID()`
        edges = self._road_network.graph.getEdges(False)
        waypoint_by_lane_memo = {}
        shape_waypoints = []

        for edge in edges:
            for lane in edge.getLanes():
                _, new_wps = self._shape_waypoints_along_lane(
                    lane, waypoint_by_lane_memo
                )
                shape_waypoints += new_wps

        return shape_waypoints

    def _shape_waypoints_along_lane(self, lane, waypoint_by_lane_memo):
        lane_queue = queue.Queue()
        lane_queue.put((lane, None))
        shape_waypoints = []
        initial_waypoint = None
        while not lane_queue.empty():
            lane, previous_wp = lane_queue.get()
            if lane.getID() in waypoint_by_lane_memo:
                if previous_wp is None:
                    continue

                previous_wp.nexts.append(waypoint_by_lane_memo[lane.getID()])
                continue

            lane_data = self._road_network.lane_data_for_lane(lane)
            lane_shape = [np.array(p) for p in lane.getShape(False)]
            right_of_way = self._does_lane_have_right_of_way(lane)

            assert len(lane_shape) >= 2, repr(lane_shape)

            heading = vec_to_radians(lane_shape[1] - lane_shape[0])
            heading = Heading(heading)

            first_waypoint = LinkedWaypoint(
                wp=Waypoint(
                    id=self._unique_waypoint_id(),
                    pos=lane_shape[0],
                    heading=heading,
                    lane_width=lane.getWidth(),
                    speed_limit=lane_data.lane_speed,
                    lane_id=lane.getID(),
                    lane_index=lane.getIndex(),
                    right_of_way=right_of_way,
                ),
                nexts=[],
            )

            if previous_wp is not None:
                previous_wp.nexts.append(first_waypoint)

            if initial_waypoint is None:
                initial_waypoint = first_waypoint

            waypoint_by_lane_memo[lane.getID()] = first_waypoint
            shape_waypoints.append(first_waypoint)
            curr_waypoint = first_waypoint

            for p1, p2 in zip(lane_shape[1:], lane_shape[2:]):
                heading_ = vec_to_radians(p2 - p1)
                heading_ = Heading(heading_)
                linked_waypoint = LinkedWaypoint(
                    wp=Waypoint(
                        id=self._unique_waypoint_id(),
                        pos=p1,
                        heading=heading_,
                        lane_width=lane.getWidth(),
                        speed_limit=lane_data.lane_speed,
                        lane_id=lane.getID(),
                        lane_index=lane.getIndex(),
                        right_of_way=right_of_way,
                    ),
                    nexts=[],
                )

                shape_waypoints.append(linked_waypoint)
                curr_waypoint.nexts.append(linked_waypoint)
                curr_waypoint = linked_waypoint

            # Add a waypoint for the last point of the current lane
            last_linked_waypoint = LinkedWaypoint(
                wp=Waypoint(
                    id=self._unique_waypoint_id(),
                    pos=lane_shape[-1],
                    heading=curr_waypoint.wp.heading,
                    lane_width=curr_waypoint.wp.lane_width,
                    speed_limit=curr_waypoint.wp.speed_limit,
                    lane_id=curr_waypoint.wp.lane_id,
                    lane_index=curr_waypoint.wp.lane_index,
                    right_of_way=right_of_way,
                ),
                nexts=[],
            )

            shape_waypoints.append(last_linked_waypoint)
            curr_waypoint.nexts.append(last_linked_waypoint)
            curr_waypoint = last_linked_waypoint

            for out_connection in lane.getOutgoing():
                out_lane = out_connection.getToLane()

                # Use internal lanes of junctions (if we're at a junction)
                via_lane_id = out_connection.getViaLaneID()
                if via_lane_id:
                    out_lane = self._road_network.lane_by_id(via_lane_id)

                lane_queue.put((out_lane, curr_waypoint))

        return initial_waypoint, shape_waypoints

    def _edge(self, wp):
        lane = self._road_network.lane_by_id(wp.lane_id)
        return lane.getEdge()
