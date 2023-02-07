import logging
from pathlib import Path
from typing import Optional
from smarts.core.coordinates import BoundingBox
from smarts.core.road_map import RoadMap
from smarts.sstudio.types import MapSpec

from av2.map.map_api import ArgoverseStaticMap
import av2.geometry.polyline_utils as polyline_utils
import av2.rendering.vector as vector_plotting_utils


class ArgoverseRoadMap(RoadMap):
    """A road map for an Argoverse 2 scenario."""

    def __init__(self, map_spec: MapSpec, avm: ArgoverseStaticMap):
        self._log = logging.getLogger(self.__class__.__name__)
        self._avm = avm
        self._argoverse_scenario_id = avm.log_id
        self._map_spec = map_spec
        # self._surfaces = dict()
        self._lanes = dict()
        self._roads = dict()
        # self._features = dict()
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

    @property
    def source(self) -> str:
        """Path to the directory containing the map JSON file."""
        return self._map_spec.source

    @property
    def bounding_box(self) -> Optional[BoundingBox]:
        """The minimum bounding box that contains the map geometry. May return `None` to indicate
        the map is unbounded.
        """
        raise NotImplementedError()

    def is_same_map(self, map_spec) -> bool:
        """Check if the MapSpec Object source points to the same RoadMap instance as the current"""
        raise NotImplementedError

    def to_glb(self, glb_dir: str):
        """Build a glb file for camera rendering and envision"""
        raise NotImplementedError()

    def lane_by_id(self, lane_id: str) -> RoadMap.Lane:
        lane = self._lanes.get(lane_id)
        assert lane, f"ArgoverseMap got request for unknown lane_id: '{lane_id}'"
        return lane

    def road_by_id(self, road_id: str) -> RoadMap.Road:
        road = self._roads.get(road_id)
        assert road, f"ArgoverseMap got request for unknown road_id: '{road_id}'"
        return road


if __name__ == "__main__":
    dataset_dir = "/home/saul/argoverse/train"
    scenario_id = "0000b0f9-99f9-4a1f-a231-5be9e4c523f7"
    source = str(Path(dataset_dir) / scenario_id)
    spec = MapSpec(source=source)
    map = ArgoverseRoadMap.from_spec(spec)
    assert map.source == source
