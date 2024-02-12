# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
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
import heapq
import logging
import random
import time
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from shapely.geometry import Point as SPoint
from shapely.geometry import Polygon

from smarts.core.coordinates import BoundingBox, Heading, Point, Pose, RefLinePoint
from smarts.core.lanepoints import LanePoint, LanePoints, LinkedLanePoint
from smarts.core.road_map import RoadMap, RoadMapWithCaches, Waypoint
from smarts.core.route_cache import RouteWithCache
from smarts.core.utils.core_math import (
    inplace_unwrap,
    line_intersect_vectorized,
    radians_to_vec,
    vec_2d,
)
from smarts.core.utils.glb import make_map_glb, make_road_line_glb
from smarts.sstudio.sstypes import MapSpec

try:
    import rtree
    from av2.geometry.interpolate import interp_arc
    from av2.map.lane_segment import LaneMarkType, LaneSegment
    from av2.map.map_api import ArgoverseStaticMap
except:
    raise ImportError(
        "Missing dependencies for Argoverse. Install them using the command `pip install -e .[argoverse]` at the source directory."
    )


class ArgoverseMap(RoadMapWithCaches):
    """A road map for an `Argoverse 2` scenario."""

    DEFAULT_LANE_SPEED = 16.67  # m/s

    LANE_MARKINGS = frozenset(
        {
            LaneMarkType.DASH_SOLID_WHITE,
            LaneMarkType.DASHED_WHITE,
            LaneMarkType.DOUBLE_SOLID_WHITE,
            LaneMarkType.DOUBLE_DASH_WHITE,
            LaneMarkType.SOLID_WHITE,
            LaneMarkType.SOLID_DASH_WHITE,
            LaneMarkType.NONE,
        }
    )

    ROAD_MARKINGS = frozenset(
        {
            LaneMarkType.DASH_SOLID_YELLOW,
            LaneMarkType.DASHED_YELLOW,
            LaneMarkType.DOUBLE_SOLID_YELLOW,
            LaneMarkType.DOUBLE_DASH_YELLOW,
            LaneMarkType.SOLID_YELLOW,
            LaneMarkType.SOLID_DASH_YELLOW,
            LaneMarkType.SOLID_BLUE,
        }
    )

    def __init__(self, map_spec: MapSpec, avm: ArgoverseStaticMap):
        super().__init__()
        self._log = logging.getLogger(self.__class__.__name__)
        self._avm = avm
        self._argoverse_scenario_id = avm.log_id
        self._map_spec = map_spec
        self._surfaces = dict()
        self._lanes: Dict[str, ArgoverseMap.Lane] = dict()
        self._roads: Dict[str, ArgoverseMap.Road] = dict()
        self._features = dict()
        self._lane_rtree = None
        self._load_map_data()
        self._waypoints_cache = ArgoverseMap._WaypointsCache()

    @classmethod
    def from_spec(cls, map_spec: MapSpec):
        """Generate a road map from the given specification."""
        scenario_dir = Path(map_spec.source)
        scenario_id = scenario_dir.stem
        map_path = scenario_dir / f"log_map_archive_{scenario_id}.json"

        if not map_path.exists():
            logging.warning(f"Map not found: {map_path}")
            return None

        avm = ArgoverseStaticMap.from_json(map_path)
        assert avm.log_id == scenario_id, "Loaded map ID does not match expected ID"
        return cls(map_spec, avm)

    def _compute_lane_intersections(self):
        intersections: Dict[str, Set[str]] = dict()

        lane_ids_todo = [lane_id for lane_id in self._lanes.keys()]

        # Build rtree
        lane_rtree = rtree.index.Index()
        lane_rtree.interleaved = True
        bboxes = dict()
        for idx, lane_id in enumerate(lane_ids_todo):
            # Using the centerline here is much faster than using the lane polygon
            lane_pts = self._lanes[lane_id]._centerline
            bbox = (
                np.amin(lane_pts[:, 0]),
                np.amin(lane_pts[:, 1]),
                np.amax(lane_pts[:, 0]),
                np.amax(lane_pts[:, 1]),
            )
            bboxes[lane_id] = bbox
            lane_rtree.add(idx, bbox)

        for lane_id in lane_ids_todo:
            lane = self._lanes[lane_id]
            lane_intersections = intersections.setdefault(lane_id, set())

            # Filter out any lanes that don't intersect this lane's bbox
            indicies = lane_rtree.intersection(bboxes[lane_id])

            # Filter out any other lanes we don't want to check against
            lanes_to_test = []
            for idx in indicies:
                cand_id = lane_ids_todo[idx]
                if cand_id == lane_id:
                    continue
                # Skip intersections we've already computed
                if cand_id in lane_intersections:
                    continue
                # ... and sub-lanes of the same original lane
                cand_lane = self._lanes[cand_id]
                # Don't check intersection with incoming/outgoing lanes
                if cand_lane in lane.incoming_lanes or cand_lane in lane.outgoing_lanes:
                    continue
                # ... or lanes in same road (TAI?)
                if lane.road == cand_lane.road:
                    continue
                lanes_to_test.append(cand_id)
            if not lanes_to_test:
                continue

            # Main loop -- check each segment of the lane polyline against the
            # polyline of each candidate lane (--> algorithm is O(l^2)
            line1 = lane._centerline
            for cand_id in lanes_to_test:
                line2 = np.array(self._lanes[cand_id]._centerline)
                C = np.roll(line2, 0, axis=0)[:-1]
                D = np.roll(line2, -1, axis=0)[:-1]
                for i in range(len(line1) - 1):
                    a = line1[i]
                    b = line1[i + 1]
                    if line_intersect_vectorized(a, b, C, D):
                        lane_intersections.add(cand_id)
                        intersections.setdefault(cand_id, set()).add(lane_id)
                        break

        for lane_id, intersect_ids in intersections.items():
            self._lanes[lane_id]._intersections = [
                self.lane_by_id(id) for id in intersect_ids
            ]

    def _load_map_data(self):
        start = time.time()

        all_ids = set(self._avm.get_scenario_lane_segment_ids())
        processed_ids = set()
        for lane_seg in self._avm.get_scenario_lane_segments():
            # If this is a rightmost lane, create a road with its neighbors
            if lane_seg.right_neighbor_id is None:
                neighbors: List[int] = []
                cur_seg = lane_seg
                while True:
                    left_mark = cur_seg.left_lane_marking.mark_type
                    left_id = cur_seg.left_neighbor_id
                    if (
                        left_id is not None
                        and left_mark in ArgoverseMap.LANE_MARKINGS
                        and left_id in self._avm.vector_lane_segments
                    ):
                        # There is a valid lane to the left, so add it and continue
                        left_seg = self._avm.vector_lane_segments[left_id]

                        # Edge case: sometimes there can be a cycle (2 lanes can have each other as their left neighbor)
                        if left_seg.left_neighbor_id == cur_seg.id:
                            break

                        cur_seg = left_seg
                        neighbors.append(left_id)
                    else:
                        break  # This is the leftmost lane in the road, so stop

                # Create the lane objects
                road_id = "road"
                lanes = []
                for index, seg_id in enumerate([lane_seg.id] + neighbors):
                    road_id += f"-{seg_id}"
                    lane_id = f"lane-{seg_id}"
                    seg = self._avm.vector_lane_segments[seg_id]
                    lane = ArgoverseMap.Lane(self, lane_id, seg, index)
                    assert lane_id not in self._lanes
                    self._lanes[lane_id] = lane
                    processed_ids.add(seg_id)
                    lanes.append(lane)

                # Create the road and fill in references
                road = ArgoverseMap.Road(road_id, lanes)
                assert road_id not in self._roads
                self._roads[road_id] = road
                for lane in lanes:
                    lane._road = road

        # Create lanes for the remaining lane segments, each with their own road
        remaining_ids = all_ids - processed_ids
        for seg_id in remaining_ids:
            lane_seg = self._avm.vector_lane_segments[seg_id]
            road_id = f"road-{lane_seg.id}"
            lane_id = f"lane-{lane_seg.id}"

            lane = ArgoverseMap.Lane(self, lane_id, lane_seg, 0)
            road = ArgoverseMap.Road(road_id, [lane])
            lane._road = road

            assert road_id not in self._roads
            assert lane_id not in self._lanes
            self._roads[road_id] = road
            self._lanes[lane_id] = lane

        # Patch in incoming/outgoing lanes now that all lanes have been created
        for lane in self._lanes.values():
            lane._incoming_lanes = [
                self.lane_by_id(f"lane-{seg_id}")
                for seg_id in lane.lane_seg.predecessors
                if seg_id in all_ids
            ]
            lane._outgoing_lanes = [
                self.lane_by_id(f"lane-{seg_id}")
                for seg_id in lane.lane_seg.successors
                if seg_id in all_ids
            ]

        self._compute_lane_intersections()

        end = time.time()
        elapsed = round((end - start) * 1000.0, 3)
        self._log.info(f"Loading Argoverse map took: {elapsed} ms")

    @property
    def source(self) -> str:
        """Path to the directory containing the map JSON file."""
        return self._map_spec.source

    @cached_property
    def bounding_box(self) -> Optional[BoundingBox]:
        xs, ys = np.array([]), np.array([])
        for lane_seg in self._avm.get_scenario_lane_segments():
            xs = np.concatenate((xs, lane_seg.polygon_boundary[:, 0]))
            ys = np.concatenate((ys, lane_seg.polygon_boundary[:, 1]))

        return BoundingBox(
            min_pt=Point(x=np.min(xs), y=np.min(ys)),
            max_pt=Point(x=np.max(xs), y=np.max(ys)),
        )

    def is_same_map(self, map_spec) -> bool:
        return map_spec.source == self._map_spec.source

    def _compute_traffic_dividers(self) -> Tuple[List, List]:
        lane_dividers = []  # divider between lanes with same traffic direction
        road_dividers = []  # divider between roads with opposite traffic direction
        processed_ids = []

        for lane_seg in self._avm.get_scenario_lane_segments():
            if lane_seg.id in processed_ids or lane_seg.is_intersection:
                continue
            if lane_seg.right_neighbor_id is None:
                cur_seg = lane_seg
                while True:
                    if (
                        cur_seg.left_neighbor_id is None
                        or cur_seg.id in processed_ids
                        or (
                            cur_seg.left_neighbor_id
                            not in self._avm.vector_lane_segments
                        )
                    ):
                        break  # This is the leftmost lane in the road, so stop
                    else:
                        left_mark = cur_seg.left_lane_marking.mark_type
                        lane = self.lane_by_id(f"lane-{cur_seg.id}")
                        left_boundary = [(p[0], p[1]) for p in lane.left_pts]

                        lane_cl = self._avm.get_lane_segment_centerline(cur_seg.id)
                        left_cl = self._avm.get_lane_segment_centerline(
                            cur_seg.left_neighbor_id
                        )
                        lane_dir = lane_cl[1] - lane_cl[0]
                        left_dir = left_cl[1] - left_cl[0]
                        angle = np.arccos(
                            np.dot(lane_dir, left_dir)
                            / (np.linalg.norm(lane_dir) * np.linalg.norm(left_dir))
                        )
                        same_dir = angle < 0.1

                        if left_mark in ArgoverseMap.LANE_MARKINGS and same_dir:
                            lane_dividers.append(left_boundary)
                        else:
                            road_dividers.append(left_boundary)
                        processed_ids.append(cur_seg.id)
                        cur_seg = self._avm.vector_lane_segments[
                            cur_seg.left_neighbor_id
                        ]

        return lane_dividers, road_dividers

    def to_glb(self, glb_dir):
        polygons = []
        for lane_id, lane in self._lanes.items():
            metadata = {
                "road_id": lane.road.road_id,
                "lane_id": lane_id,
                "lane_index": lane.index,
            }
            polygons.append((lane.shape(), metadata))

        lane_dividers, edge_dividers = self._compute_traffic_dividers()

        map_glb = make_map_glb(
            polygons, self.bounding_box, lane_dividers, edge_dividers
        )
        map_glb.write_glb(Path(glb_dir) / "map.glb")

        road_lines_glb = make_road_line_glb(edge_dividers)
        road_lines_glb.write_glb(Path(glb_dir) / "road_lines.glb")

        lane_lines_glb = make_road_line_glb(lane_dividers)
        lane_lines_glb.write_glb(Path(glb_dir) / "lane_lines.glb")

    class Surface(RoadMapWithCaches.Surface):
        """Surface representation for `Argoverse` maps."""

        def __init__(self, surface_id: str, road_map):
            self._surface_id = surface_id

        @property
        def surface_id(self) -> str:
            return self._surface_id

        @property
        def is_drivable(self) -> bool:
            return True

    def surface_by_id(self, surface_id: str) -> Optional[RoadMap.Surface]:
        return self._surfaces.get(surface_id)

    class Lane(RoadMapWithCaches.Lane, Surface):
        """Lane representation for `Argoverse` maps."""

        def __init__(
            self, map: "ArgoverseMap", lane_id: str, lane_seg: LaneSegment, index: int
        ):
            super().__init__(lane_id, map)
            self._map = map
            self._lane_id = lane_id
            self.lane_seg = lane_seg
            self._index = index
            self._road = None
            self._incoming_lanes = None
            self._outgoing_lanes = None
            self._intersections = None

            self._polygon = lane_seg.polygon_boundary[:, :2]
            self._centerline = self._map._avm.get_lane_segment_centerline(lane_seg.id)[
                :, :2
            ]

            xs = self._polygon[:, 0]
            ys = self._polygon[:, 1]
            self._bbox = BoundingBox(
                min_pt=Point(x=np.amin(xs), y=np.amin(ys)),
                max_pt=Point(x=np.amax(xs), y=np.amax(ys)),
            )

            # Compute equally-spaced points for lane boundaries by interpolating
            n = len(self._centerline)
            self.left_pts = interp_arc(
                n, points=self.lane_seg.left_lane_boundary.xyz[:, :2]
            )
            self.right_pts = interp_arc(
                n, points=self.lane_seg.right_lane_boundary.xyz[:, :2]
            )

        def __hash__(self) -> int:
            return hash(self.lane_id)

        @property
        def bounding_box(self):
            return self._bbox

        @property
        def lane_id(self) -> str:
            return self._lane_id

        @property
        def road(self) -> RoadMap.Road:
            return self._road

        @property
        def speed_limit(self) -> Optional[float]:
            return ArgoverseMap.DEFAULT_LANE_SPEED

        @lru_cache(maxsize=1024)
        def width_at_offset(self, lane_point_s: float) -> Tuple[float, float]:
            world_point = self.from_lane_coord(
                RefLinePoint(lane_point_s, 0)
            ).as_np_array[:2]
            deltas = self._centerline - world_point
            dists = np.linalg.norm(deltas, axis=1)
            closest_index = np.argmin(dists)
            p1 = self.left_pts[closest_index]
            p2 = self.right_pts[closest_index]
            width = np.linalg.norm(np.subtract(p2, p1))
            return width, 1.0

        @cached_property
        def length(self) -> float:
            length = 0
            for p1, p2 in zip(self._centerline, self._centerline[1:]):
                length += np.linalg.norm(np.subtract(p2, p1))
            return length

        @cached_property
        def center_polyline(self) -> List[Point]:
            return [Point(p[0], p[1]) for p in self._centerline]

        @property
        def in_junction(self) -> bool:
            return self.lane_seg.is_intersection

        @property
        def index(self) -> int:
            return self._index

        @lru_cache(maxsize=4)
        def shape(
            self, buffer_width: float = 0.0, default_width: Optional[float] = None
        ) -> Polygon:
            return Polygon(self._polygon)

        @cached_property
        def lanes_in_same_direction(self) -> List[RoadMap.Lane]:
            return [lane for lane in self.road.lanes if lane.lane_id != self.lane_id]

        @cached_property
        def lane_to_left(self) -> Tuple[Optional[RoadMap.Lane], bool]:
            result = None
            for other in self.lanes_in_same_direction:
                if other.index > self.index and (
                    not result or other.index < result.index
                ):
                    result = other
            return result, True

        @cached_property
        def lane_to_right(self) -> Tuple[Optional[RoadMap.Lane], bool]:
            result = None
            for other in self.lanes_in_same_direction:
                if other.index < self.index and (
                    not result or other.index > result.index
                ):
                    result = other
            return result, True

        @property
        def incoming_lanes(self) -> List[RoadMap.Lane]:
            return self._incoming_lanes

        @property
        def outgoing_lanes(self) -> List[RoadMap.Lane]:
            return self._outgoing_lanes

        @lru_cache(maxsize=16)
        def oncoming_lanes_at_offset(self, offset: float) -> List[RoadMap.Lane]:
            result = []
            radius = 1.1 * self.width_at_offset(offset)[0]
            pt = self.from_lane_coord(RefLinePoint(offset))
            nearby_lanes = self._map.nearest_lanes(pt, radius=radius)
            if not nearby_lanes:
                return result
            my_vect = self.vector_at_offset(offset)
            my_norm = np.linalg.norm(my_vect)
            if my_norm == 0:
                return result
            threshold = -0.995562  # cos(175*pi/180)
            for lane, _ in nearby_lanes:
                if lane == self:
                    continue
                lane_refline_pt = lane.to_lane_coord(pt)
                lv = lane.vector_at_offset(lane_refline_pt.s)
                lv_norm = np.linalg.norm(lv)
                if lv_norm == 0:
                    continue
                lane_angle = np.dot(my_vect, lv) / (my_norm * lv_norm)
                if lane_angle < threshold:
                    result.append(lane)
            return result

        @cached_property
        def foes(self) -> List[RoadMap.Lane]:
            foes = set(self._intersections)
            foes |= {
                incoming
                for outgoing in self.outgoing_lanes
                for incoming in outgoing.incoming_lanes
                if incoming != self
            }
            return list(foes)

        @lru_cache(maxsize=8)
        def contains_point(self, point: Point) -> bool:
            assert type(point) == Point
            if (
                self._bbox.min_pt.x <= point[0] <= self._bbox.max_pt.x
                and self._bbox.min_pt.y <= point[1] <= self._bbox.max_pt.y
            ):
                return self.shape().contains(point.as_shapely)
            return False

        def waypoint_paths_for_pose(
            self, pose: Pose, lookahead: int, route: Optional[RoadMap.Route] = None
        ) -> List[List[Waypoint]]:
            if not self.is_drivable:
                return []
            road_ids = [road.road_id for road in route.roads] if route else None
            return self._waypoint_paths_at(pose.point, lookahead, road_ids)

        def waypoint_paths_at_offset(
            self,
            offset: float,
            lookahead: int = 30,
            route: Optional[RoadMap.Route] = None,
        ) -> List[List[Waypoint]]:
            if not self.is_drivable:
                return []
            wp_start = self.from_lane_coord(RefLinePoint(offset))
            road_ids = [road.road_id for road in route.roads] if route else None
            return self._waypoint_paths_at(wp_start, lookahead, road_ids)

        def _waypoint_paths_at(
            self,
            point: Point,
            lookahead: int,
            filter_road_ids: Optional[Sequence[str]] = None,
        ) -> List[List[Waypoint]]:
            if not self.is_drivable:
                return []
            closest_linked_lp = (
                self._map._lanepoints.closest_linked_lanepoint_on_lane_to_point(
                    point, self._lane_id
                )
            )
            return self._map._waypoints_starting_at_lanepoint(
                closest_linked_lp,
                lookahead,
                tuple(filter_road_ids) if filter_road_ids else (),
                point,
            )

    def lane_by_id(self, lane_id: str) -> "ArgoverseMap.Lane":
        lane = self._lanes.get(lane_id)
        assert lane, f"ArgoverseMap got request for unknown lane_id: '{lane_id}'"
        return lane

    def _build_lane_r_tree(self):
        result = rtree.index.Index()
        result.interleaved = True
        for idx, lane in enumerate(self._lanes.values()):
            xs = lane._polygon[:, 0]
            ys = lane._polygon[:, 1]
            bounding_box = (
                np.amin(xs),
                np.amin(ys),
                np.amax(xs),
                np.amax(ys),
            )
            result.add(idx, bounding_box)
        return result

    def _get_neighboring_lanes(
        self, x: float, y: float, r: float = 0.1
    ) -> List[Tuple[RoadMapWithCaches.Lane, float]]:
        neighboring_lanes = []
        if self._lane_rtree is None:
            self._lane_rtree = self._build_lane_r_tree()

        spt = SPoint(x, y)
        lanes = list(self._lanes.values())
        for i in self._lane_rtree.intersection((x - r, y - r, x + r, y + r)):
            lane = lanes[i]
            d = lane.shape().distance(spt)
            if d < r:
                neighboring_lanes.append((lane, d))
        return neighboring_lanes

    @lru_cache(maxsize=1024)
    def nearest_lanes(
        self,
        point: Point,
        radius: Optional[float] = None,
        include_junctions: bool = False,
    ) -> List[Tuple[RoadMapWithCaches.Lane, float]]:
        if radius is None:
            radius = 5
        candidate_lanes = self._get_neighboring_lanes(point[0], point[1], r=radius)
        candidate_lanes.sort(key=lambda lane_dist_tup: lane_dist_tup[1])
        return candidate_lanes

    @lru_cache(maxsize=16)
    def road_with_point(
        self,
        point: Point,
        *,
        lanes_to_search: Optional[Sequence["RoadMap.Lane"]] = None,
    ) -> Optional[RoadMap.Road]:
        # Lookup nearest lanes if no search lanes were provided
        if not lanes_to_search:
            radius = 5
            lanes = [nl for (nl, _) in self.nearest_lanes(point, radius)]
        else:
            lanes = lanes_to_search

        for lane in lanes:
            if lane.contains_point(point):
                return lane.road
        return None

    class Road(RoadMapWithCaches.Road, Surface):
        """Road representation for `Argoverse` maps."""

        def __init__(self, road_id: str, lanes: List[RoadMap.Lane]):
            super().__init__(road_id, None)
            self._road_id = road_id
            self._lanes = lanes

            x_mins, y_mins, x_maxs, y_maxs = [], [], [], []
            for lane in self._lanes:
                # pytype: disable=attribute-error
                x_mins.append(lane.bounding_box.min_pt.x)
                y_mins.append(lane.bounding_box.min_pt.y)
                x_maxs.append(lane.bounding_box.max_pt.x)
                y_maxs.append(lane.bounding_box.max_pt.y)
                # pytype: enable=attribute-error

            self._bbox = BoundingBox(
                min_pt=Point(x=min(x_mins), y=min(y_mins)),
                max_pt=Point(x=max(x_maxs), y=max(y_maxs)),
            )

        def __hash__(self) -> int:
            return hash(self.road_id)

        @property
        def road_id(self) -> str:
            return self._road_id

        @property
        def composite_road(self) -> RoadMap.Road:
            """Return an abstract Road composed of one or more RoadMap.Road segments
            (including this one) that has been inferred to correspond to one continuous
            real-world road.  May return same object as self."""
            return self

        @property
        def is_composite(self) -> bool:
            """Returns True if this Road object was inferred
            and composed out of subordinate Road objects."""
            return False

        @cached_property
        def is_junction(self) -> bool:
            for lane in self.lanes:
                if lane.foes or len(lane.incoming_lanes) > 1:
                    return True
            return False

        @cached_property
        def length(self) -> float:
            # Neighbouring lanes in Argoverse can be different lengths. Since this is
            # just used for routes, we take the average lane length in this road.
            return sum([lane.length for lane in self.lanes]) / len(self.lanes)

        @property
        def incoming_roads(self) -> List[RoadMap.Road]:
            return list(
                {in_lane.road for lane in self.lanes for in_lane in lane.incoming_lanes}
            )

        @property
        def outgoing_roads(self) -> List[RoadMap.Road]:
            return list(
                {
                    out_lane.road
                    for lane in self.lanes
                    for out_lane in lane.outgoing_lanes
                }
            )

        @lru_cache(maxsize=16)
        def oncoming_roads_at_point(self, point: Point) -> List[RoadMap.Road]:
            result = []
            for lane in self.lanes:
                offset = lane.to_lane_coord(point).s
                result += [
                    ol.road
                    for ol in lane.oncoming_lanes_at_offset(offset)
                    if ol.road != self
                ]
            return result

        @property
        def parallel_roads(self) -> List[RoadMap.Road]:
            return []

        @property
        def lanes(self) -> List[RoadMap.Lane]:
            return self._lanes

        def lane_at_index(self, index: int) -> RoadMap.Lane:
            return self.lanes[index]

        @lru_cache(maxsize=8)
        def contains_point(self, point: Point) -> bool:
            if (
                self._bbox.min_pt.x <= point[0] <= self._bbox.max_pt.x
                and self._bbox.min_pt.y <= point[1] <= self._bbox.max_pt.y
            ):
                for lane in self._lanes:
                    if lane.contains_point(point):
                        return True
            return False

        @lru_cache(maxsize=4)
        def shape(
            self, buffer_width: float = 0.0, default_width: Optional[float] = None
        ) -> Polygon:
            leftmost_lane = self.lane_at_index(len(self.lanes) - 1)
            rightmost_lane = self.lane_at_index(0)
            left_pts = leftmost_lane.lane_seg.left_lane_boundary.xyz[:, :2]
            right_pts = rightmost_lane.lane_seg.right_lane_boundary.xyz[:, :2]
            polygon_pts = np.concatenate(
                (left_pts, right_pts[::-1], np.array([left_pts[0]]))
            )
            return Polygon(polygon_pts)

    def road_by_id(self, road_id: str) -> RoadMap.Road:
        road = self._roads.get(road_id)
        assert (
            road
        ), f"{ArgoverseMap.__name__} got request for unknown road_id: '{road_id}'"
        return road

    class Route(RouteWithCache):
        """Describes a route between `Argoverse` roads."""

        def __init__(
            self,
            road_map: RoadMap,
            roads: List[RoadMap.Road] = [],
            start_lane: Optional[RoadMap.Lane] = None,
            end_lane: Optional[RoadMap.Lane] = None,
        ):
            super().__init__(road_map, start_lane, end_lane)
            self._roads = roads
            self._length = sum([road.length for road in roads])

        @property
        def roads(self) -> List[RoadMap.Road]:
            return self._roads

        @property
        def road_length(self) -> float:
            return self._length

        @cached_property
        def geometry(self) -> Sequence[Sequence[Tuple[float, float]]]:
            return [list(road.shape(0.0).exterior.coords) for road in self.roads]

    @staticmethod
    def _shortest_route(start: RoadMap.Road, end: RoadMap.Road) -> List[RoadMap.Road]:
        queue = [(start.length, start.road_id, start)]
        came_from = dict()
        came_from[start] = None
        cost_so_far = dict()
        cost_so_far[start] = start.length
        current: Optional[RoadMap.Road] = None

        # Dijkstra’s Algorithm
        while queue:
            (_, _, current) = heapq.heappop(queue)
            if current == end:
                break
            for out_road in current.outgoing_roads:
                new_cost = cost_so_far[current] + out_road.length
                if out_road not in cost_so_far or new_cost < cost_so_far[out_road]:
                    cost_so_far[out_road] = new_cost
                    came_from[out_road] = current
                    heapq.heappush(queue, (new_cost, out_road.road_id, out_road))

        # This means we couldn't find a valid route since the queue is empty
        if current != end:
            return []

        # Reconstruct path
        current = end
        path = []
        while current != start:
            if current is not None:
                path.append(current)
            current = came_from[current]
        path.append(start)
        path.reverse()
        return path

    def _generate_routes(
        self,
        start_road: RoadMap.Road,
        start_lane: RoadMap.Lane,
        end_road: RoadMap.Road,
        end_lane: RoadMap.Lane,
        via: Optional[Sequence[RoadMap.Road]],
        max_to_gen: int,
    ) -> List[RoadMap.Route]:
        assert (
            max_to_gen == 1
        ), "multiple route generation not yet supported for Argoverse"

        roads = [start_road]
        if via:
            roads += via
        if end_road != start_road:
            roads.append(end_road)

        route_roads = []
        for cur_road, next_road in zip(roads, roads[1:] + [None]):
            if not next_road:
                route_roads.append(cur_road)
                break
            sub_route = ArgoverseMap._shortest_route(cur_road, next_road) or []
            if len(sub_route) < 2:
                self._log.warning(
                    f"Unable to find valid path between {(cur_road.road_id, next_road.road_id)}."
                )
                return [ArgoverseMap.Route(road_map=self)]
            # The sub route includes the boundary roads (cur_road, next_road).
            # We clip the latter to prevent duplicates
            route_roads.extend(sub_route[:-1])

        return [
            ArgoverseMap.Route(
                road_map=self,
                roads=route_roads,
                start_lane=start_lane,
                end_lane=end_lane,
            )
        ]

    def random_route(
        self,
        max_route_len: int = 10,
        starting_road: Optional[RoadMap.Road] = None,
        only_drivable: bool = True,
    ) -> RoadMap.Route:
        assert not starting_road or not only_drivable or starting_road.is_drivable
        next_roads = [starting_road] if starting_road else list(self._roads.values())
        route_roads = []
        if only_drivable:
            next_roads = [r for r in next_roads if r.is_drivable]
        while next_roads and len(route_roads) < max_route_len:
            cur_road = random.choice(next_roads)
            route_roads.append(cur_road)
            next_roads = list(cur_road.outgoing_roads)
        return ArgoverseMap.Route(road_map=self, roads=route_roads)

    def empty_route(self) -> RoadMap.Route:
        return ArgoverseMap.Route(self)

    def route_from_road_ids(
        self, road_ids: Sequence[str], resolve_intermediaries: bool = False
    ) -> RoadMap.Route:
        return ArgoverseMap.Route.from_road_ids(self, road_ids, resolve_intermediaries)

    class _WaypointsCache:
        def __init__(self):
            self.lookahead = 0
            self.point = Point(0, 0)
            self.filter_road_ids = ()
            self._starts = {}

        # XXX:  all vehicles share this cache now (as opposed to before
        # when it was in Plan.py and each vehicle had its own cache).
        # TODO: probably need to add vehicle_id to the key somehow (or just make it bigger)
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
            point: Point,
            filter_road_ids: tuple,
            llp,
            paths: List[List[Waypoint]],
        ):
            """Update the current cache if not already cached."""
            if not self._match(lookahead, point, filter_road_ids):
                self.lookahead = lookahead
                self.point = point
                self.filter_road_ids = filter_road_ids
                self._starts = {}
            self._starts[llp.lp.lane.lane_id] = paths

        def query(
            self,
            lookahead: int,
            point: Point,
            filter_road_ids: tuple,
            llp,
        ) -> Optional[List[List[Waypoint]]]:
            """Attempt to find previously cached waypoints"""
            if self._match(lookahead, point, filter_road_ids):
                hit = self._starts.get(llp.lp.lane.lane_id, None)
                if hit:
                    # consider just returning all of them (not slicing)?
                    return [path[: (lookahead + 1)] for path in hit]
                return None

    @cached_property
    def _lanepoints(self):
        assert self._map_spec.lanepoint_spacing > 0
        return LanePoints.from_argoverse(self, spacing=self._map_spec.lanepoint_spacing)

    def _resolve_in_junction(self, junction_lane: RoadMap.Lane) -> List[RoadMap.Lane]:
        # There are no paths we can trace back through the junction, so return
        if len(junction_lane.road.incoming_roads) == 0:
            return []

        # Trace back to the road that leads into the junction
        inc_road: RoadMap.Road = junction_lane.road.incoming_roads[0]
        lanes = []
        for out_road in inc_road.outgoing_roads:
            lanes.extend([lane for lane in out_road.lanes])
        return lanes

    def waypoint_paths(
        self,
        pose: Pose,
        lookahead: int,
        within_radius: float = 5,
        route: Optional[RoadMap.Route] = None,
    ) -> List[List[Waypoint]]:
        road_ids = []
        if route and route.roads:
            road_ids = [road.road_id for road in route.roads]
        if road_ids:
            return self._waypoint_paths_along_route(pose.point, lookahead, road_ids)

        # No route provided, so generate paths based on the closest lanepoints
        waypoint_paths = []
        closest_lps: List[LanePoint] = self._lanepoints.closest_lanepoints(
            pose, maximum_count=15
        )
        closest_lane: RoadMap.Lane = closest_lps[0].lane

        # First, see if we are in a junction and need to do something special
        if closest_lane.in_junction:
            # Get the set of all nearby junction lanes with similar heading
            junction_lanes: Set[RoadMap.Lane] = set()
            for lp in closest_lps:
                rel_heading = lp.pose.heading.relative_to(pose.heading)
                if lp.lane.in_junction and abs(rel_heading) < np.pi / 2:
                    junction_lanes.add(lp.lane)

            # Get set of all lanes leading through the junction
            wp_lanes = set()
            for junction_lane in junction_lanes:
                wp_lanes = wp_lanes.union(set(self._resolve_in_junction(junction_lane)))

            # Generate waypoints for each junction lane
            for wp_lane in wp_lanes:
                new_paths = [
                    path
                    for path in wp_lane._waypoint_paths_at(pose.point, lookahead)
                    if path[0].lane_id == wp_lane.lane_id
                ]
                for path in new_paths:
                    if (
                        len(path) > 0
                        and np.linalg.norm(np.subtract(path[0].pos, pose.position[:2]))
                        < 8
                        and abs(path[0].heading.relative_to(pose.heading)) < np.pi / 3
                    ):
                        waypoint_paths.append(path)

        # Otherwise, just generate waypoints for the closest lane
        if len(waypoint_paths) < 1:
            for lane in closest_lane.road.lanes:
                waypoint_paths += lane._waypoint_paths_at(pose.point, lookahead)

        result = sorted(waypoint_paths, key=lambda p: p[0].lane_id)
        assert len(result) > 0, "Waypoint paths should not be empty"
        return result

    def _waypoint_paths_along_route(
        self, point: Point, lookahead: int, route: Sequence[str]
    ) -> List[List[Waypoint]]:
        """finds the closest lane to vehicle's position that is on its route,
        then gets waypoint paths from all lanes in its road there."""
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
            waypoint_paths += lane._waypoint_paths_at(point, lookahead, route)

        return sorted(waypoint_paths, key=len, reverse=True)

    @staticmethod
    def _equally_spaced_path(
        path: Sequence[LinkedLanePoint],
        point: Point,
        lp_spacing: float,
    ) -> List[Waypoint]:
        """given a list of LanePoints starting near point, return corresponding
        Waypoints that may not be evenly spaced (due to lane change) but start at point.
        """

        continuous_variables = [
            "positions_x",
            "positions_y",
            "headings",
            "lane_width",
            "speed_limit",
            "lane_offset",
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

            ref_lanepoints_coordinates["lane_width"].append(lanepoint.lp.lane_width)

            ref_lanepoints_coordinates["lane_offset"].append(
                lanepoint.lp.lane.offset_along_lane(lanepoint.lp.pose.point)
            )

            ref_lanepoints_coordinates["speed_limit"].append(
                lanepoint.lp.lane.speed_limit
            )

        ref_lanepoints_coordinates["headings"] = inplace_unwrap(
            ref_lanepoints_coordinates["headings"]
        )
        first_lp_heading = ref_lanepoints_coordinates["headings"][0]
        lp_position = path[0].lp.pose.point.as_np_array[:2]
        vehicle_pos = point.as_np_array[:2]
        heading_vec = radians_to_vec(first_lp_heading)
        projected_distant_lp_vehicle = np.inner(
            (vehicle_pos - lp_position), heading_vec
        )

        ref_lanepoints_coordinates["positions_x"][0] = (
            lp_position[0] + projected_distant_lp_vehicle * heading_vec[0]
        )
        ref_lanepoints_coordinates["positions_y"][0] = (
            lp_position[1] + projected_distant_lp_vehicle * heading_vec[1]
        )

        cumulative_path_dist = np.cumsum(
            np.sqrt(
                np.ediff1d(ref_lanepoints_coordinates["positions_x"], to_begin=0) ** 2
                + np.ediff1d(ref_lanepoints_coordinates["positions_y"], to_begin=0) ** 2
            )
        )

        if len(cumulative_path_dist) <= lp_spacing:
            lp = path[0].lp

            return [
                Waypoint(
                    pos=lp.pose.position[:2],
                    heading=lp.pose.heading,
                    lane_width=lp.lane.width_at_offset(0)[0],
                    speed_limit=lp.lane.speed_limit,
                    lane_id=lp.lane.lane_id,
                    lane_index=lp.lane.index,
                    lane_offset=lp.lane.offset_along_lane(lp.pose.point),
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

        waypoint_path = []
        for idx in range(len(path)):
            waypoint_path.append(
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
                    lane_offset=evenly_spaced_coordinates["lane_offset"][idx],
                )
            )

        return waypoint_path

    def _waypoints_starting_at_lanepoint(
        self,
        lanepoint: LinkedLanePoint,
        lookahead: int,
        filter_road_ids: tuple,
        point: Point,
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
            ArgoverseMap._equally_spaced_path(
                path,
                point,
                self._map_spec.lanepoint_spacing,
            )
            for path in lanepoint_paths
        ]

        self._waypoints_cache.update(
            lookahead, point, filter_road_ids, lanepoint, result
        )

        return result
