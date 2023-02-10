from functools import lru_cache
import logging
import math
from pathlib import Path
from cached_property import cached_property
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from smarts.core.coordinates import BoundingBox, Point, Pose
from smarts.core.road_map import RoadMap
from smarts.core.utils.glb import make_map_glb
from smarts.sstudio.types import MapSpec
from av2.map.map_api import ArgoverseStaticMap
from av2.map.lane_segment import LaneMarkType
import av2.geometry.polyline_utils as polyline_utils
import av2.rendering.vector as vector_plotting_utils
from shapely.geometry import Polygon


class ArgoverseMap(RoadMap):
    """A road map for an Argoverse 2 scenario."""

    def __init__(self, map_spec: MapSpec, avm: ArgoverseStaticMap):
        self._log = logging.getLogger(self.__class__.__name__)
        self._log.setLevel(logging.INFO)
        self._avm = avm
        self._argoverse_scenario_id = avm.log_id
        self._map_spec = map_spec
        self._surfaces = dict()
        self._lanes: Dict[str, ArgoverseMap.Lane] = dict()
        self._roads: Dict[str, ArgoverseMap.Road] = dict()
        self._features = dict()
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
                    if cur_seg.left_neighbor_id is not None and left == LaneMarkType.DASHED_WHITE:
                        # There is a valid lane to the left, so add it and continue
                        neighbours.append(cur_seg.left_neighbor_id)
                        cur_seg = self._avm.vector_lane_segments[cur_seg.left_neighbor_id]
                    else:
                        break  # This is the leftmost lane in the road, so stop

                # Create the lane objects
                road_id = "road"
                lanes = []
                for index, seg_id in enumerate([lane_seg.id] + neighbours):
                    road_id += f"-{seg_id}"
                    lane_id = f"lane-{seg_id}"
                    seg = self._avm.vector_lane_segments[seg_id]
                    lane = ArgoverseMap.Lane(lane_id, seg.polygon_boundary[:, :2], index)
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

            lane = ArgoverseMap.Lane(lane_id, lane_seg.polygon_boundary[:, :2], 0)
            road = ArgoverseMap.Road(road_id, [lane])
            lane._road = road

            assert road_id not in self._roads
            assert lane_id not in self._lanes
            self._roads[road_id] = road
            self._lanes[lane_id] = lane

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
        """Check if the MapSpec Object source points to the same RoadMap instance as the current"""
        raise NotImplementedError()

    def _compute_lane_polygons(self):
        polygons = []
        for lane_id, lane in self._lanes.items():
            metadata = {
                "road_id": lane.road.road_id,
                "lane_id": lane_id,
                # "lane_index": lane.index, TODO
            }
            polygons.append((lane.shape(), metadata))
        return polygons

    def to_glb(self, glb_dir):
        polygons = self._compute_lane_polygons()
        # lane_dividers, edge_dividers = self._compute_traffic_dividers()

        map_glb = make_map_glb(polygons, self.bounding_box, [], [])
        map_glb.write_glb(Path(glb_dir) / "map.glb")

        # road_lines_glb = self._make_road_line_glb(edge_dividers)
        # road_lines_glb.write_glb(Path(glb_dir) / "road_lines.glb")

        # lane_lines_glb = self._make_road_line_glb(lane_dividers)
        # lane_lines_glb.write_glb(Path(glb_dir) / "lane_lines.glb")

    class Surface(RoadMap.Surface):
        def __init__(self, surface_id: str):
            self._surface_id = surface_id

        @property
        def surface_id(self) -> str:
            return self._surface_id

        @property
        def is_drivable(self) -> bool:
            return True

    def surface_by_id(self, surface_id: str) -> RoadMap.Surface:
        return self._surfaces.get(surface_id)

    class Lane(RoadMap.Lane, Surface):
        def __init__(self, lane_id: str, polygon, index: int):
            super().__init__(lane_id)
            self._lane_id = lane_id
            self._polygon = polygon
            self._index = index
            self._road = None

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
            return None

        @property
        def length(self) -> float:
            """The length of this lane."""
            raise NotImplementedError()

        @property
        def in_junction(self) -> bool:
            """If this lane is a part of a junction (usually an intersection.)"""
            raise NotImplementedError()

        @property
        def index(self) -> int:
            """when not in_junction, 0 is outer / right-most (relative to lane heading) lane on road.
            otherwise, index scheme is implementation-dependent, but must be deterministic."""
            # TAI:  UK roads
            raise NotImplementedError()

        @property
        def lanes_in_same_direction(self) -> List[RoadMap.Lane]:
            """returns all other lanes on this road where traffic goes
            in the same direction.  it is currently assumed these will be
            adjacent to one another.  In junctions, diverging lanes
            should not be included."""
            raise NotImplementedError()

        @property
        def lane_to_left(self) -> Tuple[RoadMap.Lane, bool]:
            """Note: left is defined as 90 degrees clockwise relative to the lane heading.
            (I.e., positive `t` in the RefLine coordinate system.)
            Second result is True if lane is in the same direction as this one
            In junctions, diverging lanes should not be included."""
            raise NotImplementedError()

        @property
        def lane_to_right(self) -> Tuple[RoadMap.Lane, bool]:
            """Note: right is defined as 90 degrees counter-clockwise relative to the lane heading.
            (I.e., negative `t` in the RefLine coordinate system.)
            Second result is True if lane is in the same direction as this one.
            In junctions, diverging lanes should not be included."""
            raise NotImplementedError()

        @property
        def incoming_lanes(self) -> List[RoadMap.Lane]:
            """Lanes leading into this lane."""
            raise NotImplementedError()

        @property
        def outgoing_lanes(self) -> List[RoadMap.Lane]:
            """Lanes leading out of this lane."""
            raise NotImplementedError()

        def oncoming_lanes_at_offset(self, offset: float) -> List[RoadMap.Lane]:
            """Returns a list of nearby lanes at offset that are (roughly)
            parallel to this one but go in the opposite direction."""
            raise NotImplementedError()

        @property
        def foes(self) -> List[RoadMap.Lane]:
            """All lanes that in some ways intersect with (cross) this one,
            including those that have the same outgoing lane as this one,
            and so might require right-of-way rules.  This should only
            ever happen in junctions."""
            raise NotImplementedError()

        @lru_cache(maxsize=4)
        def shape(self, buffer_width: float = 0.0, default_width: Optional[float] = None) -> Polygon:
            return Polygon(self._polygon)

    def lane_by_id(self, lane_id: str) -> RoadMap.Lane:
        lane = self._lanes.get(lane_id)
        assert lane, f"ArgoverseMap got request for unknown lane_id: '{lane_id}'"
        return lane

    class Road(RoadMap.Road, Surface):
        def __init__(self, road_id: str, lanes: List[RoadMap.Lane]):
            super().__init__(road_id)
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
            """Note that a junction can be an intersection ('+') or a 'T', 'Y', 'L', etc."""
            raise NotImplementedError()

        @property
        def length(self) -> float:
            """The length of this road."""
            raise NotImplementedError()

        @property
        def incoming_roads(self) -> List[RoadMap.Road]:
            """All roads that lead into this road."""
            raise NotImplementedError()

        @property
        def outgoing_roads(self) -> List[RoadMap.Road]:
            """All roads that lead out of this road."""
            raise NotImplementedError()

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
            """Gets the lane with the given index."""
            raise NotImplementedError()

    def road_by_id(self, road_id: str) -> RoadMap.Road:
        road = self._roads.get(road_id)
        assert road, f"ArgoverseMap got request for unknown road_id: '{road_id}'"
        return road
