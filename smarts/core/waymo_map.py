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
import os
import rtree
from functools import lru_cache
from subprocess import check_output
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
from waymo_open_dataset.protos import scenario_pb2
from enum import Enum
import numpy as np
from cached_property import cached_property

from smarts.sstudio.types import MapSpec

from .coordinates import BoundingBox, Heading, Point, Pose, RefLinePoint
from .lanepoints import LanePoints, LinkedLanePoint
from .road_map import RoadMap, Waypoint
from .utils.file import read_tfrecord_file
from .utils.math import offset_along_shape, position_at_shape_offset, distance_point_to_polygon


class WaymoMap(RoadMap):
    """A map associated with a Waymo dataset"""
    DEFAULT_LANE_SPEED = 16.67  # in m/s
    DEFAULT_LANE_WIDTH = 3.2

    def __init__(self, map_spec: MapSpec, scenario):
        self._log = logging.getLogger(self.__class__.__name__)
        self._map_spec = map_spec
        self._surfaces = {}
        self._lanes: Dict[int, RoadMap.Lane] = {}
        self._roads: Dict[int, RoadMap.Road] = {}
        self._map_features = {}
        self._default_lane_width = WaymoMap.DEFAULT_LANE_WIDTH
        self._lane_rtree = None
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
        for i in range(len(scenario.map_features)):
            map_feature = scenario.map_features[i]
            key = map_feature.WhichOneof("feature_data")
            if key is not None:
                self._map_features.setdefault(key, []).append(
                    (getattr(map_feature, key), map_feature.id)
                )

        for lane_feat, lane_id in self._map_features["lane"]:
            lane = WaymoMap.Lane(self, lane_id, lane_feat)
            self._lanes[lane_id] = lane
            self._surfaces[lane_id] = lane

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
        def __init__(self, road_map: RoadMap, lane_id: str, lane_feat):
            super().__init__(lane_id, road_map)
            self._map = road_map
            self._lane_id = lane_id
            self._lane_feat = lane_feat
            self._lane_pts = [np.array([p.x, p.y]) for p in lane_feat.polyline]

            if lane_feat.speed_limit_mph:
                self._speed_limit = lane_feat.speed_limit_mph * 0.44704
            else:
                self._speed_limit = WaymoMap.DEFAULT_LANE_SPEED

            self._lane_width = None
            self._bounding_box = None

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

        @lru_cache(maxsize=8)
        def width_at_offset(self, lane_point_s: float) -> Tuple[float, float]:
            return self._lane_width, 1.0

        @lru_cache(maxsize=8)
        def from_lane_coord(self, lane_point: RefLinePoint) -> Point:
            x, y = position_at_shape_offset(self._lane_pts, lane_point.s)
            return Point(x=x, y=y)

        @lru_cache(maxsize=8)
        def to_lane_coord(self, world_point: Point) -> RefLinePoint:
            return super().to_lane_coord(world_point)

        @lru_cache(maxsize=8)
        def center_at_point(self, point: Point) -> Point:
            return super().center_at_point(point)

        @lru_cache(8)
        def vector_at_offset(self, start_offset: float) -> np.ndarray:
            return super().vector_at_offset(start_offset)

        @lru_cache(maxsize=8)
        def center_pose_at_point(self, point: Point) -> Pose:
            return super().center_pose_at_point(point)

        @lru_cache(maxsize=8)
        def curvature_radius_at_offset(
                self, offset: float, lookahead: int = 5
        ) -> float:
            return super().curvature_radius_at_offset(offset, lookahead)

        @cached_property
        def bounding_box(self) -> List[Tuple[float, float]]:
            """Get the minimal axis aligned bounding box that contains all geometry in this lane."""
            # XXX: This signature is wrong. It should return Optional[BoundingBox]
            x_coordinates, y_coordinates = zip(*self.lane_polygon)
            self._bounding_box = [
                (min(x_coordinates), min(y_coordinates)),
                (max(x_coordinates), max(y_coordinates)),
            ]
            return self._bounding_box

        @lru_cache(maxsize=8)
        def contains_point(self, point: Point) -> bool:
            if (
                    self._bounding_box[0][0] <= point[0] <= self._bounding_box[1][0]
                    and self._bounding_box[0][1] <= point[1] <= self._bounding_box[1][1]
            ):
                lane_point = self.to_lane_coord(point)
                return (
                        abs(lane_point.t) <= (self._lane_width / 2)
                        and 0 <= lane_point.s < self.length
                )
            return False

        @lru_cache(maxsize=8)
        def project_along(
                self, start_offset: float, distance: float
        ) -> Set[Tuple[RoadMap.Lane, float]]:
            return super().project_along(start_offset, distance)

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

    def _build_lane_r_tree(self):
        result = rtree.index.Index()
        result.interleaved = True
        all_lanes = list(self._lanes.values())
        for idx, lane in enumerate(all_lanes):
            bounding_box = (
                lane.bounding_box[0][0],
                lane.bounding_box[0][1],
                lane.bounding_box[1][0],
                lane.bounding_box[1][1],
            )
            result.add(idx, bounding_box)
        return result

    def _get_neighboring_lanes(
        self, x: float, y: float, r: float = 0.1
    ) -> List[Tuple[RoadMap.Lane, float]]:
        neighboring_lanes = []
        all_lanes = list(self._lanes.values())
        if self._lane_rtree is None:
            self._lane_rtree = self._build_lane_r_tree()

        for i in self._lane_rtree.intersection((x - r, y - r, x + r, y + r)):
            lane = all_lanes[i]
            d = distance_point_to_polygon((x, y), lane.lane_polygon)
            if d < r:
                neighboring_lanes.append((lane, d))
        return neighboring_lanes

    @lru_cache(maxsize=16)
    def nearest_lanes(
        self, point: Point, radius: Optional[float] = None, include_junctions=False
    ) -> List[Tuple[RoadMap.Lane, float]]:
        if radius is None:
            radius = max(10, 2 * self._default_lane_width)
        candidate_lanes = self._get_neighboring_lanes(point[0], point[1], r=radius)
        candidate_lanes.sort(key=lambda lane_dist_tup: lane_dist_tup[1])
        return candidate_lanes

    def nearest_lane(
        self, point: Point, radius: float = None, include_junctions=False
    ) -> Optional[RoadMap.Lane]:
        nearest_lanes = self.nearest_lanes(point, radius, include_junctions)
        return nearest_lanes[0][0] if nearest_lanes else None
