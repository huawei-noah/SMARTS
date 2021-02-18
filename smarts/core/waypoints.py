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
import warnings
from collections import defaultdict, namedtuple
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.interpolate import interp1d

from smarts.core.utils.file import supress_pkg_resources

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", "numpy.ufunc size changed")
    with supress_pkg_resources():
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
class Waypoint:
    pos: np.ndarray  # Point positioned on center of lane
    heading: Heading  # Heading angle of lane at this point (radians)
    lane_width: float  # Width of lane at this point (meters)
    speed_limit: float  # Lane speed in m/s
    lane_id: str  # ID of lane under waypoint
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
        if not isinstance(other, Waypoint):
            return False
        return (
            self.pos.all() == other.pos.all()
            and self.heading == other.heading
            and self.lane_width == other.lane_width
            and self.speed_limit == other.speed_limit
            and self.lane_id == other.lane_id
            and self.lane_index == other.lane_index
        )


LinkedWaypoint = namedtuple(
    "LinkedWaypoint",
    [
        "wp",  # Waypoint: current waypoint
        "nexts",  # list of LinkedWaypoint: list of next immediate waypoints
        # it's a list of waypoints because a path may branch at junctions
        "is_shape_wp",
    ],
    defaults=[None, [], False],
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

    def closest_waypoint_batched(self, poses, filter_from_count, within_radius):
        linked_waypoints = self._closest_linked_wp_in_kd_tree_with_pose_batched(
            poses,
            self._linked_waypoints,
            self._waypoints_kd_tree,
            within_radius=within_radius,
            k=filter_from_count,
        )

        return [l_wps[0].wp for l_wps in linked_waypoints]

    def closest_waypoint(self, pose, filter_from_count=10, within_radius=10):
        return self.closest_waypoint_batched(
            [pose], filter_from_count, within_radius=within_radius
        )[0]

    def closest_waypoints(self, point, desired_count):
        linked_waypoints = self._closest_linked_wp_in_kd_tree_with_pose_batched(
            [point],
            self._linked_waypoints,
            self._waypoints_kd_tree,
            within_radius=10,
            k=desired_count,
            keep_all_k=True,
        )[0]

        return [lwp.wp for lwp in linked_waypoints]

    def closest_waypoint_on_lane(self, pose, lane_id):
        lane_kd_tree = self._waypoints_kd_tree_by_lane_id[lane_id]
        linked_waypoint = self._closest_linked_wp_in_kd_tree_with_pose_batched(
            [pose],
            self._waypoints_by_lane_id[lane_id],
            lane_kd_tree,
            within_radius=10,
            k=10,
        )[0][0]

        return linked_waypoint.wp

    def closest_waypoint_on_lane_to_point(self, position, lane_id):
        lane_kd_tree = self._waypoints_kd_tree_by_lane_id[lane_id]
        linked_waypoint = self._closest_linked_wp_in_kd_tree_batched(
            [position], self._waypoints_by_lane_id[lane_id], lane_kd_tree
        )[0][0]

        return linked_waypoint.wp

    def waypoint_paths_on_lane_at(
        self, point, lane_id, lookahead, filter_edge_ids: Sequence[str] = None
    ):
        lane_kd_tree = self._waypoints_kd_tree_by_lane_id[lane_id]
        closest_linked_wp = self._closest_linked_wp_in_kd_tree_batched(
            [point], self._waypoints_by_lane_id[lane_id], lane_kd_tree, k=1
        )[0][0]

        unlinked_waypoint_paths = self._waypoints_starting_at_waypoint(
            closest_linked_wp, lookahead, point, filter_edge_ids
        )

        return unlinked_waypoint_paths

    def waypoint_paths_at(self, pose, lookahead, filter_from_count=3, within_radius=5):
        closest_linked_wp = self.closest_waypoint(
            pose, filter_from_count=filter_from_count, within_radius=within_radius
        )
        closest_lane = self._road_network.lane_by_id(closest_linked_wp.lane_id)

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

    def _closest_linked_wp_in_kd_tree_with_pose_batched(
        self, poses, waypoints, tree, within_radius, k=10, keep_all_k=False
    ):
        linked_waypoints = self._closest_linked_wp_in_kd_tree_batched(
            [pose.position[:2] for pose in poses], waypoints, tree, k=k
        )

        radius_sq = within_radius * within_radius
        linked_waypoints = [
            sorted(
                l_wps,
                key=lambda _lwp: squared_dist(poses[idx].position[:2], _lwp.wp.pos),
            )
            for idx, l_wps in enumerate(linked_waypoints)
        ]
        # exclude those outside radius except closest
        if not keep_all_k:
            linked_waypoints = [
                [
                    _lwp
                    for i, _lwp in enumerate(_lwps)
                    if squared_dist(poses[idx].position[:2], _lwp.wp.pos) <= radius_sq
                    or i == 0
                ]
                for idx, _lwps in enumerate(linked_waypoints)
            ]
        # Get the nearest point for the points where the radius check failed
        unfound_waypoints = [
            (i, poses[i]) for i, group in enumerate(linked_waypoints) if len(group) == 0
        ]
        if len(unfound_waypoints) > 0:
            remaining_linked_wps = self._closest_linked_wp_in_kd_tree_batched(
                [pose.position[:2] for _, pose in unfound_waypoints],
                waypoints,
                tree=tree,
                k=k,
            )
            # Replace the empty waypoint locations
            for (i, _), wps in [
                g for g in zip(unfound_waypoints, remaining_linked_wps)
            ]:
                linked_waypoints[i] = [wps]

        return [
            sorted(
                l_wps,
                key=lambda _lwp: squared_dist(poses[idx].position[:2], _lwp.wp.pos)
                + abs(poses[idx].heading.relative_to(_lwp.wp.heading)),
            )
            for idx, l_wps in enumerate(linked_waypoints)
        ]

    def _closest_linked_wp_in_kd_tree_batched(self, points, linked_wps, tree, k=1):
        p2ds = np.array([vec_2d(p) for p in points])
        closest_indices = tree.query(
            p2ds, k=min(k, len(linked_wps)), return_distance=False, sort_results=True
        )

        return [[linked_wps[idx] for idx in idxs] for idxs in closest_indices]

    def _waypoints_starting_at_waypoint(
        self,
        waypoint: LinkedWaypoint,
        lookahead,
        point,
        filter_edge_ids: Sequence[str] = None,
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

        return [self._equally_spaced_path(path, point) for path in waypoint_paths]

    def _equally_spaced_path(self, path, point):
        continuous_variables = [
            "ref_wp_positions_x",
            "ref_wp_positions_y",
            "ref_wp_headings",
            "ref_wp_lane_width",
            "ref_wp_speed_limit",
        ]

        discrete_variables = ["ref_wp_lane_id", "ref_wp_lane_index"]

        ref_waypoints_coordinates = {
            parameter: [] for parameter in (continuous_variables + discrete_variables)
        }
        for idx, waypoint in enumerate(path):
            if not waypoint.is_shape_wp and 0 < idx < len(path) - 1:
                continue
            ref_waypoints_coordinates["ref_wp_positions_x"].append(waypoint.wp.pos[0])
            ref_waypoints_coordinates["ref_wp_positions_y"].append(waypoint.wp.pos[1])
            ref_waypoints_coordinates["ref_wp_headings"].append(
                waypoint.wp.heading.as_bullet
            )
            ref_waypoints_coordinates["ref_wp_lane_id"].append(waypoint.wp.lane_id)
            ref_waypoints_coordinates["ref_wp_lane_index"].append(
                waypoint.wp.lane_index
            )
            ref_waypoints_coordinates["ref_wp_lane_width"].append(
                waypoint.wp.lane_width
            )
            ref_waypoints_coordinates["ref_wp_speed_limit"].append(
                waypoint.wp.speed_limit
            )

        ref_waypoints_coordinates["ref_wp_headings"] = np.unwrap(
            ref_waypoints_coordinates["ref_wp_headings"]
        )
        first_wp_heading = ref_waypoints_coordinates["ref_wp_headings"][0]
        wp_position = np.array([*path[0].wp.pos, 0])
        vehicle_pos = np.array([point[0], point[1], 0])
        heading_vector = np.array(
            [
                *radians_to_vec(first_wp_heading),
                0,
            ]
        )
        projected_distant_wp_vehicle = np.inner(
            (vehicle_pos - wp_position), heading_vector
        )

        ref_waypoints_coordinates["ref_wp_positions_x"][0] = (
            wp_position[0] + projected_distant_wp_vehicle * heading_vector[0]
        )
        ref_waypoints_coordinates["ref_wp_positions_y"][0] = (
            wp_position[1] + projected_distant_wp_vehicle * heading_vector[1]
        )
        # To ensure that the distance between waypoints are equal, we used
        # interpolation approach inspired by:
        # https://stackoverflow.com/a/51515357
        cumulative_path_dist = np.cumsum(
            np.sqrt(
                np.ediff1d(ref_waypoints_coordinates["ref_wp_positions_x"], to_begin=0)
                ** 2
                + np.ediff1d(
                    ref_waypoints_coordinates["ref_wp_positions_y"], to_begin=0
                )
                ** 2
            )
        )

        if len(cumulative_path_dist) <= 1:
            return [path[0].wp]

        evenly_spaced_cumulative_path_dist = np.linspace(
            0, cumulative_path_dist[-1], len(path)
        )

        evenly_spaced_coordinates = {}
        for variable in continuous_variables:
            evenly_spaced_coordinates[variable] = interp1d(
                cumulative_path_dist, ref_waypoints_coordinates[variable]
            )(evenly_spaced_cumulative_path_dist)

        for variable in discrete_variables:
            ref_coordinates = ref_waypoints_coordinates[variable]
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
        for idx, waypoint in enumerate(path):
            equally_spaced_path.append(
                Waypoint(
                    pos=np.array(
                        [
                            evenly_spaced_coordinates["ref_wp_positions_x"][idx],
                            evenly_spaced_coordinates["ref_wp_positions_y"][idx],
                        ]
                    ),
                    heading=Heading(evenly_spaced_coordinates["ref_wp_headings"][idx]),
                    lane_width=evenly_spaced_coordinates["ref_wp_lane_width"][idx],
                    speed_limit=evenly_spaced_coordinates["ref_wp_speed_limit"][idx],
                    lane_id=evenly_spaced_coordinates["ref_wp_lane_id"][idx],
                    lane_index=evenly_spaced_coordinates["ref_wp_lane_index"][idx],
                )
            )

        return equally_spaced_path

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
            if shape_wp.wp in interp_memo:
                if previous_wp is None:
                    continue
                previous_wp.nexts.append(interp_memo[shape_wp.wp])
                continue

            first_linked_waypoint = LinkedWaypoint(
                wp=Waypoint(
                    pos=shape_wp.wp.pos,
                    heading=shape_wp.wp.heading,
                    lane_width=shape_wp.wp.lane_width,
                    speed_limit=shape_wp.wp.speed_limit,
                    lane_id=shape_wp.wp.lane_id,
                    lane_index=shape_wp.wp.lane_index,
                ),
                nexts=[],
                is_shape_wp=True,
            )

            if previous_wp is not None:
                previous_wp.nexts.append(first_linked_waypoint)

            if initial_waypoint is None:
                initial_waypoint = first_linked_waypoint

            interp_memo[shape_wp.wp] = first_linked_waypoint

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

            linked_waypoint = LinkedWaypoint(
                wp=Waypoint(
                    pos=pos,
                    heading=Heading(heading),
                    lane_width=lane_width,
                    speed_limit=speed_limit,
                    lane_id=lane_id,
                    lane_index=shape_wp.wp.lane_index,
                ),
                nexts=[],
                is_shape_wp=False,
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

            assert len(lane_shape) >= 2, repr(lane_shape)

            heading = vec_to_radians(lane_shape[1] - lane_shape[0])
            heading = Heading(heading)

            first_waypoint = LinkedWaypoint(
                wp=Waypoint(
                    pos=lane_shape[0],
                    heading=heading,
                    lane_width=lane.getWidth(),
                    speed_limit=lane_data.lane_speed,
                    lane_id=lane.getID(),
                    lane_index=lane.getIndex(),
                ),
                nexts=[],
                is_shape_wp=True,
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
                        pos=p1,
                        heading=heading_,
                        lane_width=lane.getWidth(),
                        speed_limit=lane_data.lane_speed,
                        lane_id=lane.getID(),
                        lane_index=lane.getIndex(),
                    ),
                    nexts=[],
                    is_shape_wp=True,
                )

                shape_waypoints.append(linked_waypoint)
                curr_waypoint.nexts.append(linked_waypoint)
                curr_waypoint = linked_waypoint

            # Add a waypoint for the last point of the current lane
            last_linked_waypoint = LinkedWaypoint(
                wp=Waypoint(
                    pos=lane_shape[-1],
                    heading=curr_waypoint.wp.heading,
                    lane_width=curr_waypoint.wp.lane_width,
                    speed_limit=curr_waypoint.wp.speed_limit,
                    lane_id=curr_waypoint.wp.lane_id,
                    lane_index=curr_waypoint.wp.lane_index,
                ),
                nexts=[],
                is_shape_wp=True,
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
