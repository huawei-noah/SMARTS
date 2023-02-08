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
import av2.geometry.polyline_utils as polyline_utils
import av2.rendering.vector as vector_plotting_utils
from shapely.geometry import Polygon


class ArgoverseMap(RoadMap):
    """A road map for an Argoverse 2 scenario."""

    def __init__(self, map_spec: MapSpec, avm: ArgoverseStaticMap):
        self._log = logging.getLogger(self.__class__.__name__)
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

        for lane_seg in self._avm.get_scenario_lane_segments():
            road_id = f"road-{lane_seg.id}"
            lane_id = f"lane-{lane_seg.id}"
            road = ArgoverseMap.Road(road_id)
            lane = ArgoverseMap.Lane(lane_id, road, lane_seg.polygon_boundary[:, :2])
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
        def __init__(self, lane_id: str, road: RoadMap.Road, polygon):
            super().__init__(lane_id)
            self._lane_id = lane_id
            self._road = road
            self._polygon = polygon

        def __hash__(self) -> int:
            return hash(self.lane_id)

        @property
        def lane_id(self) -> str:
            return self._lane_id

        @property
        def road(self) -> RoadMap.Road:
            return self._road

        @cached_property
        def speed_limit(self) -> Optional[float]:
            raise NotImplementedError()

        @cached_property
        def length(self) -> float:
            raise NotImplementedError()

        @lru_cache(maxsize=4)
        def shape(self, buffer_width: float = 0.0, default_width: Optional[float] = None) -> Polygon:
            return Polygon(self._polygon)

    def lane_by_id(self, lane_id: str) -> RoadMap.Lane:
        lane = self._lanes.get(lane_id)
        assert lane, f"ArgoverseMap got request for unknown lane_id: '{lane_id}'"
        return lane

    class Road(RoadMap.Road, Surface):
        def __init__(self, road_id: str):
            super().__init__(road_id)
            self._road_id = road_id

        def __hash__(self) -> int:
            return hash(self.road_id)

        @property
        def road_id(self) -> str:
            return self._road_id

    def road_by_id(self, road_id: str) -> RoadMap.Road:
        road = self._roads.get(road_id)
        assert road, f"ArgoverseMap got request for unknown road_id: '{road_id}'"
        return road
