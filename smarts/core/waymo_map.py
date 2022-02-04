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


class WaymoMap(RoadMap):
    """A map associated with a Waymo dataset"""

    def __init__(self, map_spec: MapSpec):
        self._log = logging.getLogger(self.__class__.__name__)
        self._map_spec = map_spec
        self._surfaces = {}
        self._lanes = {}
        self._roads = {}
        # some keys in _map_features include:  lane, road_line, road_edge, stop_sign, crosswalk, speed_bump
        self._map_features = {}
        self._scenario_id = None
        # self._waypoints_cache = TODO
        self._lanepoints = None
        if map_spec.lanepoint_spacing is not None:
            assert map_spec.lanepoint_spacing > 0
            # XXX: this should be last here since LanePoints() calls road_network methods immediately
            self._lanepoints = LanePoints.from_waymo(
                self, spacing=map_spec.lanepoint_spacing
            )

    def _load(self, path: str, scenario_id):
        dataset = read_tfrecord_file(path)
        for record in dataset:
            parsed_scenario = scenario_pb2.Scenario()
            parsed_scenario.ParseFromString(bytearray(record))
            if parsed_scenario.scenario_id == scenario_id:
                scenario = parsed_scenario
                break
        else:
            errmsg = f"Dataset file does not contain scenario with id: {scenario_id}"
            raise ValueError(errmsg)

        self._scenario_id = scenario.scenario_id
        for i in range(len(scenario.map_features)):
            map_feature = scenario.map_features[i]
            key = map_feature.WhichOneof("feature_data")
            if key is not None:
                self._map_features.setdefault(key, []).append(
                    (getattr(map_feature, key), map_feature.id)
                )

        for lane_feats in self._map_features["lane"]:
            lane_feat = lane_feats[0]
            lane_id = lane_feats[1]
            lane = WaymoMap.Lane(self, lane_id, lane_feat)
            self._lanes[lane_id] = lane
            self._surfaces[lane_id] = lane

    @classmethod
    def from_spec(cls, map_spec: MapSpec):
        """Generate a road network from the given map specification."""
        # TODO:  check that map_spec.source encodes valid path as well as scenario_id
        pass  # TODO

    @property
    def source(self) -> str:
        return self._scenario_id  # TAI: include path too?

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
        def __init__(self, road_map, lane_id: str, lane_feat):
            super().__init__(lane_id, road_map)
            self._lane_id = lane_id
            self._map = road_map
            # since lane_feat is kept in self._map_features too,
            # we can just keep a reference to it here and extract things
            # from it on-demand
            self_lane_feat = lane_feat

            # TODO: remove below and only do the following on demand (cache if necessary)...
            self._lane_pts = [np.array([p.x, p.y]) for p in lane_feat.polyline]
            self._left_boundaries = WaymoMap.Lane._extract_boundaries(
                lane_feat.left_boundaries
            )
            self._right_boundaries = WaymoMap.Lane._extract_boundaries(
                lane_feat.right_boundaries
            )
            self._left_neighbors = WaymoMap.Lane._extract_neighbors(
                lane_feat.left_neighbors
            )
            self._right_neighbors = WaymoMap.Lane._extract_neighbors(
                lane_feat.right_neighbors
            )

        @staticmethod
        def _extract_neighbors(neighbors: Sequence) -> List[Dict[str, Any]]:
            return [
                {
                    "id": nb.feature_id,
                    "indexes": [
                        nb.self_start_index,
                        nb.self_end_index,
                        nb.neighbor_start_index,
                        nb.neighbor_end_index,
                    ],
                    "boundaries": WaymoMap.Lane._extract_boundaries(nb.boundaries),
                }
                for nb in neighbors
            ]

        @staticmethod
        def _extract_boundaries(boundaries: Sequence) -> List[Dict[str, Any]]:
            return [
                {
                    "index": [bd.lane_start_index, bd.lane_end_index],
                    "type": bd.boundary_type,
                    "id": bd.boundary_feature_id,
                }
                for bd in boundaries
            ]

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
                for exit_lanes in self._lane_feat._exit_lanes
            ]

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
