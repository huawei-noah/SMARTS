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
from functools import lru_cache
from subprocess import check_output
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
from waymo_open_dataset.protos import scenario_pb2
from enum import Enum
import numpy as np
from cached_property import cached_property
from shapely.geometry import Polygon

from smarts.sstudio.types import MapSpec

from .coordinates import BoundingBox, Heading, Point, Pose, RefLinePoint
from .lanepoints import LanePoints, LinkedLanePoint
from .road_map import RoadMap, Waypoint
from .utils.file import read_tfrecord_file
from .utils.math import offset_along_shape

# TODO: move to math or utils module
def line_intersect(a, b, c, d) -> Union[np.ndarray, None]:
    r = b - a
    s = d - c
    d = r[0] * s[1] - r[1] * s[0]

    if d == 0:
        return None

    u = ((c[0] - a[0]) * r[1] - (c[1] - a[1]) * r[0]) / d
    t = ((c[0] - a[0]) * s[1] - (c[1] - a[1]) * s[0]) / d

    if 0 <= u and u <= 1 and 0 <= t and t <= 1:
        return a + t * r

    return None


# TODO: move to math or utils module
def ray_boundary_intersect(ray_start, ray_end, boundary_pts) -> Union[np.ndarray, None]:
    for j in range(len(boundary_pts) - 1):
        b0 = boundary_pts[j]
        b1 = boundary_pts[j + 1]
        intersect_pt = line_intersect(b0, b1, ray_start, ray_end)
        if intersect_pt is not None:
            return intersect_pt
    return None


class WaymoMap(RoadMap):
    """A map associated with a Waymo dataset"""

    DEFAULT_LANE_SPEED = 16.67  # in m/s

    def __init__(self, map_spec: MapSpec, scenario):
        self._log = logging.getLogger(self.__class__.__name__)
        self._map_spec = map_spec
        self._surfaces = {}
        self._lanes: Dict[str, WaymoMap.Lane] = {}
        self._roads: Dict[str, WaymoMap.Road] = {}
        self._load_from_scenario(scenario)

        # self._waypoints_cache = TODO
        self._lanepoints = None
        if map_spec.lanepoint_spacing is not None:
            assert map_spec.lanepoint_spacing > 0
            # XXX: this should be last here since LanePoints() calls road_network methods immediately
            self._lanepoints = LanePoints.from_waymo(
                self, spacing=map_spec.lanepoint_spacing
            )

    def _load_from_scenario(self, scenario):
        features: Dict[int, Any] = {}
        lanes: List[Tuple[str, Any]] = []
        for i in range(len(scenario.map_features)):
            map_feature = scenario.map_features[i]
            key = map_feature.WhichOneof("feature_data")
            if key is not None:
                features[map_feature.id] = getattr(map_feature, key)
                if key == "lane":
                    lanes.append((str(map_feature.id), getattr(map_feature, key)))

        start = time.time()

        # First pass -- create lane objects, initial geometry computations
        for lane_id, lane_feat in lanes:
            lane = WaymoMap.Lane(self, lane_id, lane_feat, features)
            self._lanes[lane_id] = lane
            self._surfaces[lane_id] = lane

        # Second pass -- try to fill in missing points from connected lanes
        should_run = True
        i = 0
        while should_run and i < 10:
            should_run = False
            i += 1
            for lane_id, lane_feat in lanes:
                lane = self._lanes[lane_id]
                if lane.fill_in_connected_points():
                    should_run = True

        # Third pass -- create polygons
        for lane_id, lane_feat in lanes:
            lane = self._lanes[lane_id]
            lane.create_polygon()

        end = time.time()
        elapsed = round((end - start) * 1000.0, 3)
        self._log.info(f"Lanes took: {elapsed} ms")

    @classmethod
    def from_spec(cls, map_spec: MapSpec):
        """Generate a road network from the given map specification."""

        # Read the dataset file and get the specified scenario
        dataset_path = map_spec.source.split("#")[0]
        scenario_id = map_spec.source.split("#")[1]
        dataset_records = read_tfrecord_file(dataset_path)
        scenario = None
        for record in dataset_records:
            parsed_scenario = scenario_pb2.Scenario()
            parsed_scenario.ParseFromString(bytearray(record))
            if parsed_scenario.scenario_id == scenario_id:
                scenario = parsed_scenario
                break
        else:
            errmsg = f"Dataset file does not contain scenario with id: {scenario_id}"
            raise ValueError(errmsg)

        assert scenario
        return cls(map_spec, scenario)

    @property
    def source(self) -> str:
        return self._map_spec.source

    def is_same_map(self, map_spec: MapSpec) -> bool:
        """Test if the road network is identical to the given map specification."""
        pass  # TODO

    @cached_property
    def bounding_box(self) -> BoundingBox:
        """Get the minimal axis aligned bounding box that contains all map geometry."""
        pass  # TODO

    @property
    def scale_factor(self) -> float:
        """Get the scale factor between the default lane width and the default lane width."""
        return 1.0  # TODO

    def to_glb(self, at_path: str):
        """Build a glb file for camera rendering and envision"""
        pass  # TODO (or not!)

    class Surface(RoadMap.Surface):
        """Describes a surface."""

        def __init__(self, surface_id: str, road_map):
            self._surface_id = surface_id
            self._map = road_map

        @property
        def surface_id(self) -> str:
            """The identifier for this surface."""
            return self._surface_id

        @property
        def is_drivable(self) -> bool:
            """If it is possible to drive on this surface."""
            return True  # TODO?

    def surface_by_id(self, surface_id: str) -> RoadMap.Surface:
        """Find a surface by its identifier."""
        return self._surfaces.get(surface_id)

    class Lane(RoadMap.Lane, Surface):
        def __init__(self, road_map: RoadMap, lane_id: str, lane_feat, features):
            super().__init__(lane_id, road_map)
            self._map = road_map
            self._lane_id = lane_id
            self._lane_feat = lane_feat
            self._lane_pts = [np.array([p.x, p.y]) for p in lane_feat.polyline]
            if lane_feat.speed_limit_mph:
                self._speed_limit = lane_feat.speed_limit_mph * 0.44704
            else:
                self._speed_limit = WaymoMap.DEFAULT_LANE_SPEED

            # Geometry
            self._n_pts = len(self._lane_pts)
            self._normals = [None] * self._n_pts
            self._left_widths = [0] * self._n_pts
            self._right_widths = [0] * self._n_pts
            self._ray_dist = 20.0
            self._polygon = None
            self._calculate_normals()
            self._raycast_boundaries(features)

        def _calculate_normals(self):
            for i in range(self._n_pts):
                p = self._lane_pts[i]
                dp = None
                if i < self._n_pts - 1:
                    dp = self._lane_pts[i + 1] - p
                else:
                    dp = p - self._lane_pts[i - 1]

                dp /= np.linalg.norm(dp)
                angle = math.pi / 2
                normal = np.array(
                    [
                        math.cos(angle) * dp[0] - math.sin(angle) * dp[1],
                        math.sin(angle) * dp[0] + math.cos(angle) * dp[1],
                    ]
                )
                self._normals[i] = normal

        def _raycast_boundaries(self, features):
            for i in range(self._n_pts):
                ray_start = self._lane_pts[i]
                normal = self._normals[i]

                if self._lane_feat.left_neighbors:
                    sign = 1.0
                    ray_end = ray_start + sign * self._ray_dist * normal
                    for n in self._lane_feat.left_neighbors:
                        if not (n.self_start_index <= i <= n.self_end_index):
                            continue
                        feature = features[n.feature_id]
                        boundary_pts = [np.array([p.x, p.y]) for p in feature.polyline]
                        intersect_pt = ray_boundary_intersect(
                            ray_start, ray_end, boundary_pts
                        )
                        if intersect_pt is not None:
                            self._left_widths[i] = np.linalg.norm(
                                intersect_pt - ray_start
                            )
                            break

                if self._lane_feat.right_neighbors:
                    sign = -1.0
                    ray_end = ray_start + sign * self._ray_dist * normal
                    for n in self._lane_feat.right_neighbors:
                        if not (n.self_start_index <= i <= n.self_end_index):
                            continue
                        feature = features[n.feature_id]
                        boundary_pts = [np.array([p.x, p.y]) for p in feature.polyline]
                        intersect_pt = ray_boundary_intersect(
                            ray_start, ray_end, boundary_pts
                        )
                        if intersect_pt is not None:
                            self._right_widths[i] = np.linalg.norm(
                                intersect_pt - ray_start
                            )
                            break

            # Sometimes lanes that overlap are considered neighbors, so filter those out
            width_threshold = 0.5
            if (
                max(self._left_widths) > width_threshold
                or max(self._right_widths) > width_threshold
            ):
                return

            for i in [0, self._n_pts - 1]:
                ray_start = self._lane_pts[i]
                normal = self._normals[i]

                if self._lane_feat.left_boundaries:
                    sign = 1.0
                    ray_end = ray_start + sign * self._ray_dist * normal
                    for boundary in self._lane_feat.left_boundaries:
                        if not (
                            boundary.lane_start_index <= i <= boundary.lane_end_index
                        ):
                            continue
                        feature = features[boundary.boundary_feature_id]
                        boundary_pts = [np.array([p.x, p.y]) for p in feature.polyline]
                        intersect_pt = ray_boundary_intersect(
                            ray_start, ray_end, boundary_pts
                        )
                        if intersect_pt is not None:
                            self._left_widths[i] = np.linalg.norm(
                                intersect_pt - ray_start
                            )
                            break

                if self._lane_feat.right_boundaries:
                    sign = -1.0
                    ray_end = ray_start + sign * self._ray_dist * normal
                    for boundary in self._lane_feat.right_boundaries:
                        if not (
                            boundary.lane_start_index <= i <= boundary.lane_end_index
                        ):
                            continue
                        feature = features[boundary.boundary_feature_id]
                        boundary_pts = [np.array([p.x, p.y]) for p in feature.polyline]
                        intersect_pt = ray_boundary_intersect(
                            ray_start, ray_end, boundary_pts
                        )
                        if intersect_pt is not None:
                            self._right_widths[i] = np.linalg.norm(
                                intersect_pt - ray_start
                            )
                            break

        def fill_in_connected_points(self) -> bool:
            filled_in_point = False
            if (
                self._left_widths[0] == 0
                and self._right_widths[0] == 0
                and self._left_widths[-1] == 0
                and self._right_widths[-1] == 0
            ):
                for n in self.incoming_lanes:
                    if n and n._left_widths[-1] != 0:
                        self._left_widths[0] = n._left_widths[-1]
                        filled_in_point = True
                        break
                for n in self.incoming_lanes:
                    if n and n._right_widths[-1] != 0:
                        self._right_widths[0] = n._right_widths[-1]
                        filled_in_point = True
                        break
                for n in self.outgoing_lanes:
                    if n and n._left_widths[0] != 0:
                        self._left_widths[-1] = n._left_widths[0]
                        filled_in_point = True
                        break
                for n in self.outgoing_lanes:
                    if n and n._right_widths[0] != 0:
                        self._right_widths[-1] = n._right_widths[0]
                        filled_in_point = True
                        break
            return filled_in_point

        def create_polygon(self):
            max_width = max(
                self._left_widths[0],
                self._left_widths[-1],
                self._right_widths[0],
                self._right_widths[-1],
            )
            max_width = 3 if max_width == 0 else max_width

            new_left_pts = [None] * self._n_pts
            new_right_pts = [None] * self._n_pts
            for i in range(self._n_pts):
                p = self._lane_pts[i]
                new_left_pts[i] = p + (max_width * self._normals[i])
                new_right_pts[i] = p + (-1.0 * max_width * self._normals[i])

            xs, ys = [], []
            for p in new_left_pts + new_right_pts[::-1] + [new_left_pts[0]]:
                if p is not None:
                    xs.append(p[0])
                    ys.append(p[1])
            self._polygon = Polygon(list(zip(xs, ys)))

        @property
        def lane_id(self) -> str:
            return self._lane_id

        @cached_property
        def length(self) -> float:
            length = 0.0
            for i in range(len(self._lane_pts) - 1):
                a = self._lane_pts[i]
                b = self._lane_pts[i + 1]
                length += np.linalg.norm(b - a)
            return length

        @cached_property
        def incoming_lanes(self) -> List[RoadMap.Lane]:
            """Lanes leading into this lane."""
            return [
                self._map.lane_by_id(entry_lane)
                for entry_lane in self._lane_feat.entry_lanes
            ]

        @cached_property
        def outgoing_lanes(self) -> List[RoadMap.Lane]:
            """Lanes leading out of this lane."""
            return [
                self._map.lane_by_id(exit_lanes)
                for exit_lanes in self._lane_feat.exit_lanes
            ]

        def speed_limit(self) -> float:
            return self._speed_limit

        @lru_cache(maxsize=8)
        def offset_along_lane(self, world_point: Point) -> float:
            return offset_along_shape(world_point[:2], self._lane_pts)

        @cached_property
        def entry_surfaces(self) -> List[RoadMap.Surface]:
            """All surfaces leading into this lane."""
            # TODO?  can a non-lane connect into a lane?
            return self.incoming_lanes

        @cached_property
        def exit_surfaces(self) -> List[RoadMap.Surface]:
            """All surfaces leading out of this lane."""
            # TODO?  can a lane exit to a non-lane?
            return self.outgoing_lanes

    def lane_by_id(self, lane_id: str) -> RoadMap.Lane:
        # note: all lanes were cached already by _load()
        return self._lanes.get(lane_id)
