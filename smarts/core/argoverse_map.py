from functools import lru_cache
import logging
import math
from pathlib import Path
import random
from cached_property import cached_property
import time
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import rtree

from smarts.core.coordinates import BoundingBox, Point, Pose, RefLinePoint
from smarts.core.road_map import RoadMap, RoadMapWithCaches, Waypoint
from smarts.core.route_cache import RouteWithCache
from smarts.core.utils.glb import make_map_glb
from smarts.core.utils.math import line_intersect_vectorized
from smarts.sstudio.types import MapSpec
from av2.map.map_api import ArgoverseStaticMap
from av2.map.lane_segment import LaneMarkType, LaneSegment
import av2.geometry.polyline_utils as polyline_utils
import av2.rendering.vector as vector_plotting_utils
from shapely.geometry import Polygon
from shapely.geometry import Point as SPoint


class ArgoverseMap(RoadMapWithCaches):
    """A road map for an Argoverse 2 scenario."""

    DEFAULT_LANE_SPEED = 16.67  # m/s

    def __init__(self, map_spec: MapSpec, avm: ArgoverseStaticMap):
        super().__init__()
        self._log = logging.getLogger(self.__class__.__name__)
        self._log.setLevel(logging.INFO)
        self._avm = avm
        self._argoverse_scenario_id = avm.log_id
        self._map_spec = map_spec
        self._surfaces = dict()
        self._lanes: Dict[str, ArgoverseMap.Lane] = dict()
        self._roads: Dict[str, ArgoverseMap.Road] = dict()
        self._features = dict()
        self._lane_rtree = None
        self._load_map_data()
        # self._waypoints_cache = SumoRoadNetwork._WaypointsCache()
        # self._lanepoints = None
        # if map_spec.lanepoint_spacing is not None:
        #     assert map_spec.lanepoint_spacing > 0
        #     # XXX: this should be last here since LanePoints() calls road_network methods immediately
        #     self._lanepoints = LanePoints.from_sumo(self, spacing=map_spec.lanepoint_spacing)

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
            # If this is a rightmost lane, create a road with its neighbours
            if lane_seg.right_neighbor_id is None:
                neighbours: List[int] = []
                cur_seg = lane_seg
                while True:
                    left = cur_seg.left_lane_marking.mark_type
                    if (
                        cur_seg.left_neighbor_id is not None
                        and left == LaneMarkType.DASHED_WHITE
                    ):
                        # There is a valid lane to the left, so add it and continue
                        neighbours.append(cur_seg.left_neighbor_id)
                        cur_seg = self._avm.vector_lane_segments[
                            cur_seg.left_neighbor_id
                        ]
                    else:
                        break  # This is the leftmost lane in the road, so stop

                # Create the lane objects
                road_id = "road"
                lanes = []
                for index, seg_id in enumerate([lane_seg.id] + neighbours):
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

    def to_glb(self, glb_dir):
        polygons = []
        for lane_id, lane in self._lanes.items():
            metadata = {
                "road_id": lane.road.road_id,
                "lane_id": lane_id,
                # "lane_index": lane.index, TODO
            }
            polygons.append((lane.shape(), metadata))

        # lane_dividers, edge_dividers = self._compute_traffic_dividers()

        # road_lines_glb = self._make_road_line_glb(edge_dividers)
        # road_lines_glb.write_glb(Path(glb_dir) / "road_lines.glb")

        # lane_lines_glb = self._make_road_line_glb(lane_dividers)
        # lane_lines_glb.write_glb(Path(glb_dir) / "lane_lines.glb")

        map_glb = make_map_glb(polygons, self.bounding_box, [], [])
        map_glb.write_glb(Path(glb_dir) / "map.glb")

    class Surface(RoadMapWithCaches.Surface):
        def __init__(self, surface_id: str, road_map):
            self._surface_id = surface_id

        @property
        def surface_id(self) -> str:
            return self._surface_id

        @property
        def is_drivable(self) -> bool:
            return True

    def surface_by_id(self, surface_id: str) -> RoadMap.Surface:
        return self._surfaces.get(surface_id)

    class Lane(RoadMapWithCaches.Lane, Surface):
        def __init__(
            self, map: "ArgoverseMap", lane_id: str, lane_seg: LaneSegment, index: int
        ):
            super().__init__(lane_id, map)
            self._map = map
            self._lane_id = lane_id
            self.lane_seg = lane_seg
            self._polygon = lane_seg.polygon_boundary[:, :2]
            self._centerline = self._map._avm.get_lane_segment_centerline(lane_seg.id)[
                :, :2
            ]
            self._index = index
            self._road = None
            self._incoming_lanes = None
            self._outgoing_lanes = None
            self._intersections = None

        def __hash__(self) -> int:
            return hash(self.lane_id)

        @property
        def lane_id(self) -> str:
            return self._lane_id

        @property
        def road(self) -> RoadMap.Road:
            return self._road

        @property
        def speed_limit(self) -> Optional[float]:
            return ArgoverseMap.DEFAULT_LANE_SPEED

        @cached_property
        def length(self) -> float:
            length = 0
            for p1, p2 in zip(self._centerline, self._centerline[1:]):
                length += np.linalg.norm(p2 - p1)
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
        def lane_to_left(self) -> Tuple[RoadMap.Lane, bool]:
            result = None
            for other in self.lanes_in_same_direction:
                if other.index > self.index and (
                    not result or other.index < result.index
                ):
                    result = other
            return result, True

        @cached_property
        def lane_to_right(self) -> Tuple[RoadMap.Lane, bool]:
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

    def lane_by_id(self, lane_id: str) -> RoadMap.Lane:
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

    class Road(RoadMapWithCaches.Road, Surface):
        def __init__(self, road_id: str, lanes: List[RoadMap.Lane]):
            super().__init__(road_id, None)
            self._road_id = road_id
            self._lanes = lanes

        def __hash__(self) -> int:
            return hash(self.road_id)

        @property
        def road_id(self) -> str:
            return self._road_id

        @property
        def type(self) -> int:
            """The type of this road."""
            raise NotImplementedError()

        @property
        def type_as_str(self) -> str:
            """The type of this road."""
            raise NotImplementedError()

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

        @property
        def is_junction(self) -> bool:
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

        def oncoming_roads_at_point(self, point: Point) -> List[RoadMap.Road]:
            """Returns a list of nearby roads to point that are (roughly)
            parallel to this one but have lanes that go in the opposite direction."""
            raise NotImplementedError()

        @property
        def parallel_roads(self) -> List[RoadMap.Road]:
            """Returns roads that start and end at the same
            point as this one."""
            raise NotImplementedError()

        @property
        def lanes(self) -> List[RoadMap.Lane]:
            return self._lanes

        def lane_at_index(self, index: int) -> RoadMap.Lane:
            return self.lanes[index]

    def road_by_id(self, road_id: str) -> RoadMap.Road:
        road = self._roads.get(road_id)
        assert road, f"ArgoverseMap got request for unknown road_id: '{road_id}'"
        return road

    class Route(RouteWithCache):
        def __init__(self, road_map):
            super().__init__(road_map)
            self._roads = []
            self._length = 0

        @property
        def roads(self) -> List[RoadMap.Road]:
            return self._roads

        @property
        def road_length(self) -> float:
            return self._length

        def add_road(self, road: RoadMap.Road):
            """Add a road to this route."""
            self._length += road.length
            self._roads.append(road)

        @cached_property
        def geometry(self) -> Sequence[Sequence[Tuple[float, float]]]:
            return [list(road.shape(0.0).exterior.coords) for road in self.roads]

    def random_route(
        self,
        max_route_len: int = 10,
        starting_road: Optional[RoadMap.Road] = None,
        only_drivable: bool = True,
    ) -> RoadMap.Route:
        assert not starting_road or not only_drivable or starting_road.is_drivable
        route = ArgoverseMap.Route(self)
        next_roads = [starting_road] if starting_road else list(self._roads.values())
        if only_drivable:
            next_roads = [r for r in next_roads if r.is_drivable]
        while next_roads and len(route.roads) < max_route_len:
            cur_road = random.choice(next_roads)
            route.add_road(cur_road)
            next_roads = list(cur_road.outgoing_roads)
        return route

    def empty_route(self) -> RoadMap.Route:
        return ArgoverseMap.Route(self)

    def route_from_road_ids(self, road_ids: Sequence[str]) -> RoadMap.Route:
        return ArgoverseMap.Route.from_road_ids(self, road_ids)
