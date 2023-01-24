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
import math
import random
import time
from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import rtree
import trimesh
from cached_property import cached_property
from shapely.geometry import Point as SPoint
from shapely.geometry import Polygon
from trimesh.exchange import gltf
from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.protos.map_pb2 import (
    Crosswalk,
    LaneCenter,
    RoadLine,
    SpeedBump,
    StopSign,
)

from smarts.sstudio.types import MapSpec
from smarts.waymo.waymo_utils import WaymoDatasetError

from .coordinates import BoundingBox, Heading, Point, Pose, RefLinePoint
from .lanepoints import LanePoints, LinkedLanePoint
from .road_map import RoadMap, RoadMapWithCaches, Waypoint
from .route_cache import RouteWithCache
from .utils.file import read_tfrecord_file
from .utils.geometry import buffered_shape, generate_meshes_from_polygons
from .utils.math import (
    inplace_unwrap,
    line_intersect_vectorized,
    radians_to_vec,
    ray_boundary_intersect,
    vec_2d,
)


def _convert_camera(camera):
    result = {
        "name": camera.name,
        "type": "perspective",
        "perspective": {
            "aspectRatio": camera.fov[0] / camera.fov[1],
            "yfov": np.radians(camera.fov[1]),
            "znear": float(camera.z_near),
            # HACK: The trimesh gltf export doesn't include a zfar which Panda3D GLB
            #       loader expects. Here we override to make loading possible.
            "zfar": float(camera.z_near + 100),
        },
    }
    return result


gltf._convert_camera = _convert_camera


class _GLBData:
    def __init__(self, bytes_):
        self._bytes = bytes_

    def write_glb(self, output_path: str):
        """Generate a geometry file."""
        with open(output_path, "wb") as f:
            f.write(self._bytes)


class WaymoMap(RoadMapWithCaches):
    """A map associated with a Waymo dataset"""

    # Default values to fall back to
    DEFAULT_LANE_SPEED = 16.67  # in m/s
    DEFAULT_LANE_WIDTH = 4

    # Avg z-coord difference between intersecting lanes
    OVERPASS_THRESHOLD = 5.0

    # For caching tfrecord data
    _tfrecord_path: Optional[str] = None
    _tfrecord_generator: Optional[Generator[bytes, None, None]] = None
    _scenario_cache: Optional[Dict[str, Any]] = None

    def __init__(self, map_spec: MapSpec, waymo_scenario):
        super().__init__()
        self._log = logging.getLogger(self.__class__.__name__)
        self._map_spec = map_spec
        self._waymo_scenario_id = waymo_scenario.scenario_id
        self._surfaces: Dict[str, WaymoMap.Surface] = dict()
        self._lanes: Dict[str, WaymoMap.Lane] = dict()
        self._roads: Dict[str, WaymoMap.Road] = dict()
        self._features: Dict[str, WaymoMap.Feature] = dict()
        self._waymo_features: Dict[int, Any] = dict()
        self._default_lane_width = WaymoMap.DEFAULT_LANE_WIDTH
        self._lane_rtree = None
        self._no_composites = False  # for debugging purposes
        self._has_overpasses = False
        self._load_from_scenario(waymo_scenario)

        self._waypoints_cache = WaymoMap._WaypointsCache()
        self._lanepoints = None
        if map_spec.lanepoint_spacing is not None:
            assert map_spec.lanepoint_spacing > 0
            # XXX: this should be last here since LanePoints() calls road_network methods immediately
            self._lanepoints = LanePoints.from_waymo(
                self, spacing=map_spec.lanepoint_spacing
            )

    def _calculate_normals(
        self, feat_id: int
    ) -> Union[List[None], Sequence[np.ndarray]]:
        pts = self._polyline_cache[feat_id][0]
        n_pts = len(pts)

        # Special case: return early to avoid division by zero
        if n_pts == 1:
            return [np.array([0, 0])]

        normals = [None] * n_pts
        for i in range(n_pts):
            p = pts[i][:2]
            if i < n_pts - 1:
                dp = pts[i + 1][:2] - p
            else:
                dp = p - pts[i - 1][:2]

            dp /= np.linalg.norm(dp)
            angle = math.pi / 2
            normal = np.array(
                [
                    math.cos(angle) * dp[0] - math.sin(angle) * dp[1],
                    math.sin(angle) * dp[0] + math.cos(angle) * dp[1],
                ]
            )
            normals[i] = normal
        return normals

    def _raycast_boundaries(
        self, lane_dict: Dict[str, Any], feat_id: int, ray_dist: float = 20.0
    ) -> Optional[Tuple[List[float], List[float]]]:
        lane_pts = self._polyline_cache[feat_id][0]
        n_pts = len(lane_pts)
        left_widths = [0] * n_pts
        right_widths = [0] * n_pts
        normals = lane_dict["_normals"]

        for i in range(n_pts):
            ray_start = lane_pts[i][:2]
            normal = normals[i]

            if lane_dict["left_neighbors"]:
                sign = 1.0
                ray_end = ray_start + sign * ray_dist * normal
                for n in lane_dict["left_neighbors"]:
                    if not (n.self_start_index <= i <= n.self_end_index):
                        continue
                    boundary_pts = self._polyline_cache[n.feature_id][0]
                    intersect_pt = ray_boundary_intersect(
                        ray_start, ray_end, boundary_pts
                    )
                    if intersect_pt is not None:
                        left_widths[i] = np.linalg.norm(intersect_pt - ray_start)
                        break

            if lane_dict["right_neighbors"]:
                sign = -1.0
                ray_end = ray_start + sign * ray_dist * normal
                for n in lane_dict["right_neighbors"]:
                    if not (n.self_start_index <= i <= n.self_end_index):
                        continue
                    boundary_pts = self._polyline_cache[n.feature_id][0]
                    intersect_pt = ray_boundary_intersect(
                        ray_start, ray_end, boundary_pts
                    )
                    if intersect_pt is not None:
                        right_widths[i] = np.linalg.norm(intersect_pt - ray_start)
                        break

        # Sometimes lanes that overlap are considered neighbors, so filter those out
        width_threshold = 0.5
        if max(left_widths) > width_threshold or max(right_widths) > width_threshold:
            return left_widths, right_widths

        for i in [0, n_pts - 1]:
            ray_start = lane_pts[i][:2]
            normal = normals[i]

            if lane_dict["left_boundaries"]:
                sign = 1.0
                ray_end = ray_start + sign * ray_dist * normal
                for boundary in lane_dict["left_boundaries"]:
                    if not (boundary.lane_start_index <= i <= boundary.lane_end_index):
                        continue
                    feature = self._waymo_features[boundary.boundary_feature_id]
                    boundary_pts = [np.array([p.x, p.y]) for p in feature.polyline]
                    intersect_pt = ray_boundary_intersect(
                        ray_start, ray_end, boundary_pts
                    )
                    if intersect_pt is not None:
                        dist = np.linalg.norm(intersect_pt - ray_start)
                        if left_widths[i] > 0:
                            left_widths[i] = min(left_widths[i], dist)
                        else:
                            left_widths[i] = dist

            if lane_dict["right_boundaries"]:
                sign = -1.0
                ray_end = ray_start + sign * ray_dist * normal
                for boundary in lane_dict["right_boundaries"]:
                    if not (boundary.lane_start_index <= i <= boundary.lane_end_index):
                        continue
                    feature = self._waymo_features[boundary.boundary_feature_id]
                    boundary_pts = [np.array([p.x, p.y]) for p in feature.polyline]
                    intersect_pt = ray_boundary_intersect(
                        ray_start, ray_end, boundary_pts
                    )
                    if intersect_pt is not None:
                        dist = np.linalg.norm(intersect_pt - ray_start)
                        if right_widths[i] > 0:
                            right_widths[i] = min(right_widths[i], dist)
                        else:
                            right_widths[i] = dist

        return left_widths, right_widths

    def _compute_lane_intersections(self, composites: bool):
        intersections: Dict[str, Set[str]] = dict()

        lane_ids_todo = [
            lane_id
            for lane_id, lane in self._lanes.items()
            if lane.is_composite == composites
        ]

        # Build rtree
        lane_rtree = rtree.index.Index()
        lane_rtree.interleaved = True
        bboxes = dict()
        for idx, lane_id in enumerate(lane_ids_todo):
            lane_pts = self._lanes[lane_id]._lane_pts
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
                if lane._feature_id == cand_lane._feature_id:
                    continue
                # Don't check intersection with incoming/outgoing lanes
                if cand_id in lane.incoming_lanes or cand_id in lane.outgoing_lanes:
                    continue
                # ... or lanes in same road (TAI?)
                if lane.road == cand_lane.road:
                    continue
                lanes_to_test.append(cand_id)
            if not lanes_to_test:
                continue

            # Main loop -- check each segment of the lane polyline against the
            # polyline of each candidate lane (--> algorithm is O(l^2)
            line1 = lane._lane_pts
            for cand_id in lanes_to_test:
                line2 = np.array(self._lanes[cand_id]._lane_pts)
                C = np.roll(line2, 0, axis=0)[:-1]
                D = np.roll(line2, -1, axis=0)[:-1]
                for i in range(len(line1) - 1):
                    a = line1[i]
                    b = line1[i + 1]
                    if line_intersect_vectorized(a, b, C, D):
                        lane_intersections.add(cand_id)
                        intersections.setdefault(cand_id, set()).add(lane_id)
                        break

        # Remove lanes that aren't true intersections
        mappings_to_remove = []
        for lane_id, intersect_ids in intersections.items():
            lane_pts = self._lanes[lane_id]._lane_pts
            z_avg = np.average(lane_pts[:, 2])
            for intersect_id in intersect_ids:
                intersect_lane_pts = self._lanes[intersect_id]._lane_pts
                intlane_z_avg = np.average(intersect_lane_pts[:, 2])

                # Remove "overpasses" that have large z-coordinate differences
                if abs(z_avg - intlane_z_avg) > WaymoMap.OVERPASS_THRESHOLD:
                    self._has_overpasses = True
                    mappings_to_remove.append((lane_id, intersect_id))
                    continue  # already removing this pair, so skip next check

                # Remove "fake" incoming/outgoing lanes that intersect by their end points
                if np.all(np.equal(lane_pts[0], intersect_lane_pts[-1])) or np.all(
                    np.equal(lane_pts[-1], intersect_lane_pts[0])
                ):
                    mappings_to_remove.append((lane_id, intersect_id))

        # Can't do this while iterating over the sets, so do as separate step
        for id1, id2 in mappings_to_remove:
            intersections[id1].discard(id2)
            intersections[id2].discard(id1)

        for lane_id, intersect_ids in intersections.items():
            self._lanes[lane_id]._intersections = intersect_ids

    @dataclass
    class _Split:
        feat_id: int
        index: int
        structural: bool

        def __hash__(self) -> int:
            return hash(self.feat_id) ^ hash(self.index) ^ hash(self.structural)

        def __eq__(self, other) -> bool:
            return self.__class__ == other.__class__ and hash(self) == hash(other)

    @dataclass
    class _LinkedSplit:
        split: "WaymoMap._Split"
        left_splits: List["WaymoMap._Split"] = field(default_factory=lambda: [])
        right_splits: List["WaymoMap._Split"] = field(default_factory=lambda: [])
        next_split: Optional["WaymoMap._LinkedSplit"] = None
        prev_split: Optional["WaymoMap._LinkedSplit"] = None
        used: bool = False

        def __hash__(self) -> int:
            return hash(self.split)

        def __eq__(self, other) -> bool:
            return self.__class__ == other.__class__ and hash(self) == hash(other)

    class _SDict(Dict[int, _LinkedSplit]):
        @cached_property
        def sorted_keys(self) -> List[int]:
            """@return the keys in ascending order; only to be used after dict contents are final"""
            # if we want to add another dependency, it would probably be better to use SortedDict...
            return sorted(self.keys())

    _FeatureSplits = Dict[int, _SDict]

    @staticmethod
    def _lane_id(feat_id: int, index: int) -> str:
        lane_id = f"{feat_id}"
        if index > 0:
            # try to keep seg_ids the same as lane ids when not doing segmentation
            lane_id += f"_{index}"
        return lane_id

    def _interpolate_split(
        self, split: _Split, neighbors: Sequence
    ) -> Optional[_Split]:
        pld = self._polyline_cache[split.feat_id][1]
        split_dist = pld[split.index] if split.index < len(pld) else pld[-1]
        # XXX:  not symmetric!
        for nb in neighbors:
            if not (nb.self_start_index < split.index < nb.self_end_index):
                continue
            start_dist = pld[nb.self_start_index]
            assert split_dist >= start_dist
            split_perc = (split_dist - start_dist) / (
                pld[nb.self_end_index] - start_dist
            )

            nb_pld = self._polyline_cache[nb.feature_id][1]
            nb_start_dist = nb_pld[nb.neighbor_start_index]
            nb_end_dist = nb_pld[nb.neighbor_end_index]
            nb_spot = nb.neighbor_start_index
            nb_split_perc = prev_nb_split_perc = 0
            while nb_spot <= nb.neighbor_end_index:
                nb_split_perc = (nb_pld[nb_spot] - nb_start_dist) / (
                    nb_end_dist - nb_start_dist
                )
                if nb_split_perc >= split_perc:
                    break
                prev_nb_split_perc = nb_split_perc
                nb_spot += 1
            if nb_split_perc - split_perc > split_perc - prev_nb_split_perc:
                nb_spot = max(nb_spot - 1, 0)
            self._log.info(
                f"interpolating split point at {nb_spot} in neighbor {nb.feature_id} of {split.feat_id} w/ split={split}"
            )
            return WaymoMap._Split(nb.feature_id, nb_spot, split.structural)
        return None

    def _find_lane_splits(self, feat_id: int) -> _SDict:
        result = WaymoMap._SDict()
        lane_feats = self._waymo_features[feat_id]
        for side in ["left", "right"]:
            for nb in getattr(lane_feats, f"{side}_neighbors"):
                split = WaymoMap._Split(feat_id, nb.self_start_index, True)
                nb_start = result.setdefault(
                    nb.self_start_index, WaymoMap._LinkedSplit(split)
                )
                getattr(nb_start, f"{side}_splits").append(
                    WaymoMap._Split(nb.feature_id, nb.neighbor_start_index, True)
                )
                split = WaymoMap._Split(feat_id, nb.self_end_index + 1, True)
                nb_end = result.setdefault(
                    nb.self_end_index + 1, WaymoMap._LinkedSplit(split)
                )
        result.setdefault(0, WaymoMap._LinkedSplit(WaymoMap._Split(feat_id, 0, True)))
        last = len(self._polyline_cache[feat_id][0])
        result.setdefault(
            last, WaymoMap._LinkedSplit(WaymoMap._Split(feat_id, last, True))
        )
        for side in ["left", "right"]:
            boundaries = getattr(lane_feats, f"{side}_boundaries")
            prev_bdry = None
            for bdry in boundaries:
                bdry_idx = bdry.lane_start_index
                if (
                    prev_bdry
                    and bdry.boundary_type != prev_bdry.boundary_type
                    and bdry_idx - prev_bdry.lane_end_index <= 1
                ):
                    split = WaymoMap._Split(feat_id, bdry_idx, False)
                    result.setdefault(bdry_idx, WaymoMap._LinkedSplit(split))
                prev_bdry = bdry
        # interpolate for any missing neighbors...
        for linked_split in result.values():
            for side in ["left", "right"]:
                neighbors = getattr(lane_feats, f"{side}_neighbors")
                nb_split = self._interpolate_split(linked_split.split, neighbors)
                if nb_split:
                    getattr(linked_split, f"{side}_splits").append(nb_split)
        return result

    def _find_splits(self) -> _FeatureSplits:
        # find splits for all lanes individually
        feat_splits: WaymoMap._FeatureSplits = dict()
        splits_stack = deque()
        for lane_feat_id in self._feat_dicts.keys():
            lane_splits = self._find_lane_splits(lane_feat_id)
            assert len(lane_splits) >= 2
            feat_splits[lane_feat_id] = lane_splits
            for ls in lane_splits.values():
                splits_stack.append(ls)
        # then propagate them left and right...
        while splits_stack:
            linked_split = splits_stack.pop()
            for side in ["left", "right"]:
                # NOTE:  lanes in intersections can have two (or more!) neighbors on a side
                for side_split in getattr(linked_split, f"{side}_splits"):
                    side_feat_id = side_split.feat_id
                    side_index = side_split.index
                    side_lsplit = feat_splits.get(side_feat_id, WaymoMap._SDict()).get(
                        side_index
                    )
                    if not side_lsplit:
                        side_lsplit = feat_splits.setdefault(
                            side_feat_id, WaymoMap._SDict()
                        ).setdefault(side_index, WaymoMap._LinkedSplit(side_split))
                        other_side = "right" if side == "left" else "left"
                        refl_split = WaymoMap._Split(
                            linked_split.split.feat_id,
                            linked_split.split.index,
                            side_split.structural,
                        )
                        getattr(side_lsplit, f"{other_side}_splits").append(refl_split)
                        side_lane_feats = self._waymo_features[side_feat_id]
                        neighbors = getattr(side_lane_feats, f"{side}_neighbors")
                        nb_split = self._interpolate_split(side_split, neighbors)
                        if nb_split:
                            getattr(side_lsplit, f"{side}_splits").append(nb_split)
                        splits_stack.append(side_lsplit)
                    elif (
                        linked_split.split.structural
                        and not side_lsplit.split.structural
                    ):
                        side_lsplit.split.structural = True
                        splits_stack.append(side_lsplit)
        return feat_splits

    def _link_splits(self, feat_splits: _FeatureSplits):
        for linked_splits in feat_splits.values():
            prev_linked_split = None
            for split_ind in linked_splits.sorted_keys:
                linked_split = linked_splits[split_ind]
                if prev_linked_split:
                    linked_split.prev_split = prev_linked_split
                    prev_linked_split.next_split = linked_split
                prev_linked_split = linked_split

    @staticmethod
    def _map_pt_to_point(map_point) -> Point:
        return Point(map_point.x, map_point.y, map_point.z)

    @staticmethod
    def _polyline_dists(polyline) -> Tuple[np.ndarray, np.ndarray]:
        lane_pts = np.array([WaymoMap._map_pt_to_point(p) for p in polyline])

        class _Accum:
            def __init__(self):
                self._d = 0.0
                self._last_pt = None

            def accum(self, pt: np.ndarray) -> float:
                """@return accumulated distance so far"""
                if self._last_pt is not None:
                    self._d += np.linalg.norm(pt - self._last_pt)
                self._last_pt = pt
                return self._d

        q = _Accum()
        dists = np.array([q.accum(pt) for pt in lane_pts])
        return lane_pts, dists

    def _create_lane_from_split(
        self, linked_split: _LinkedSplit, feat_splits: _FeatureSplits
    ) -> "WaymoMap.Lane":
        feat_id = linked_split.split.feat_id
        feat_dict = self._feat_dicts[feat_id]
        orig_polyline = self._polyline_cache[feat_id][0]
        next_split_pt = linked_split.next_split.split.index

        lane_dict = {}
        lane_dict["type"] = feat_dict["type"]
        lane_dict["speed_limit_mph"] = feat_dict["speed_limit_mph"]
        lane_dict["interpolating"] = feat_dict["interpolating"]
        lane_dict["_normals"] = [
            np
            for i, np in enumerate(feat_dict["_normals"])
            if linked_split.split.index <= i <= next_split_pt
        ]
        lane_dict["_feature_id"] = feat_id
        lane_dict["lane_width"] = feat_dict["lane_width"]
        lane_dict["polyline"] = [
            pt
            for i, pt in enumerate(orig_polyline)
            if linked_split.split.index <= i <= next_split_pt
        ]

        if linked_split.split.index > 0:
            lane_dict["incoming_lane_ids"] = [
                WaymoMap._lane_id(feat_id, linked_split.prev_split.split.index)
            ]
        else:
            # XXX: there ought to be a better way than this!!
            incoming = []
            for el in feat_dict["entry_lanes"]:
                entry_max = len(self._polyline_cache[el][0]) - 1
                for i in reversed(feat_splits[el].sorted_keys):
                    if i < entry_max:
                        break
                else:
                    if entry_max == 0:
                        self._log.warning(
                            f"ignoring 'entry_lane' feature={el} for feature={feat_id} as it only has a single point in its polyline."
                        )
                        continue
                    i = 0
                incoming.append(WaymoMap._lane_id(el, i))
            lane_dict["incoming_lane_ids"] = incoming
        if next_split_pt < len(orig_polyline) - 1:
            lane_dict["outgoing_lane_ids"] = [WaymoMap._lane_id(feat_id, next_split_pt)]
        else:
            lane_dict["outgoing_lane_ids"] = [
                WaymoMap._lane_id(xl, 0) for xl in feat_dict["exit_lanes"]
            ]
        lane_dict["lane_to_left_info"] = linked_split.left_splits
        lane_dict["lane_to_right_info"] = linked_split.right_splits

        lane_id = WaymoMap._lane_id(feat_id, linked_split.split.index)
        lane = WaymoMap.Lane(self, lane_id, lane_dict)
        self._lanes[lane_id] = lane
        self._surfaces[lane_id] = lane
        linked_split.used = True
        return lane

    def _add_right_lanes(
        self,
        linked_split: _LinkedSplit,
        lanes: List["WaymoMap.Lane"],
        feat_splits: _FeatureSplits,
        seen: Set[_LinkedSplit],
    ) -> Tuple[bool, bool]:
        structural_split = linked_split.split.structural
        # if there's more than one lane adjacent to this at the same point, it's in a junction
        seen.add(linked_split)
        in_junction = (
            len(linked_split.right_splits) > 1 or len(linked_split.left_splits) > 1
        )
        for rt_split in linked_split.right_splits:
            rfeat = feat_splits[rt_split.feat_id]
            rt_lsplit = rfeat[rt_split.index]
            if (
                not rt_lsplit.next_split
                or rt_lsplit.split.index >= rfeat.sorted_keys[-1] - 1
                or rt_lsplit.used
                or rt_lsplit in seen
            ):
                continue
            rt_structural, rt_in_junction = self._add_right_lanes(
                rt_lsplit, lanes, feat_splits, seen
            )
            in_junction = in_junction or rt_in_junction
            structural_split = structural_split or rt_structural
        lane = self._create_lane_from_split(linked_split, feat_splits)
        lanes.append(lane)
        return structural_split, in_junction

    def _add_left_lanes(
        self,
        linked_split: _LinkedSplit,
        lanes: List["WaymoMap.Lane"],
        feat_splits: _FeatureSplits,
    ) -> Tuple[bool, bool]:
        structural_split = linked_split.split.structural
        # if there's more than one lane adjacent to this at the same point, it's in a junction
        in_junction = (
            len(linked_split.left_splits) > 1 or len(linked_split.right_splits) > 1
        )
        used = []
        for lft_split in linked_split.left_splits:
            lfeat = feat_splits[lft_split.feat_id]
            lft_lsplit = lfeat[lft_split.index]
            if (
                not lft_lsplit.next_split
                or lft_lsplit.split.index >= lfeat.sorted_keys[-1] - 1
                or lft_lsplit.used
            ):
                continue
            used.append(lft_split)
            lane = self._create_lane_from_split(lft_lsplit, feat_splits)
            lanes.append(lane)
        for lft_split in used:
            lfeat = feat_splits[lft_split.feat_id]
            lft_lsplit = lfeat[lft_split.index]
            lft_structural, lft_in_junction = self._add_left_lanes(
                lft_lsplit, lanes, feat_splits
            )
            in_junction = in_junction or lft_in_junction
            structural_split = structural_split or lft_structural
        return structural_split, in_junction

    def _create_road_from_lanes(
        self, lanes: Sequence["WaymoMap.Lane"], junction: bool
    ) -> "WaymoMap.Road":
        road = WaymoMap.Road(self, lanes, junction)
        assert road.road_id not in self._roads, f"duplicate road_id={road.road_id}"
        self._roads[road.road_id] = road
        self._surfaces[road.road_id] = road
        return road

    def _create_composite(self, composite_roads: Sequence["WaymoMap.Road"]):
        assert len(composite_roads) > 1
        composite_lanes = []
        for li in range(len(composite_roads[0].lanes)):
            lane_dict = {}
            composite_lane_id = "waymo_composite_lane:"
            for road in composite_roads:
                composite_lane_id += f":{road.lanes[li].lane_id}"
            for road in composite_roads:
                cl = road.lanes[li]
                if not lane_dict:
                    lane_dict = deepcopy(cl._lane_dict)
                else:
                    lane_dict["polyline"] += cl._lane_dict["polyline"]
                    lane_dict["_normals"] += cl._lane_dict["_normals"]
                lane_dict.setdefault("sublanes", []).append(cl.lane_id)
                assert "composite" not in cl._lane_dict
                cl._lane_dict["composite"] = composite_lane_id
            lane_dict["incoming_lane_ids"] = (
                composite_roads[0].lanes[li]._lane_dict["incoming_lane_ids"]
            )
            lane_dict["outgoing_lane_ids"] = (
                composite_roads[-1].lanes[li]._lane_dict["outgoing_lane_ids"]
            )
            lane = WaymoMap.Lane(self, composite_lane_id, lane_dict)
            self._lanes[composite_lane_id] = lane
            self._surfaces[composite_lane_id] = lane
            composite_lanes.append(lane)
        for i, cl in enumerate(composite_lanes):
            cl._lane_dict["lane_to_left_info"] = (
                composite_lanes[i + 1].lane_id if i + 1 < len(composite_lanes) else None
            )
            cl._lane_dict["lane_to_right_info"] = (
                composite_lanes[i - 1].lane_id if i > 0 else None
            )
        self._create_road_from_lanes(composite_lanes, False)

    @staticmethod
    def _can_merge_roads(
        road: "WaymoMap.Road", prev_roads: Sequence["WaymoMap.Road"]
    ) -> bool:
        if not prev_roads:
            return False
        if len(road.lanes) != len(prev_roads[-1].lanes):
            return False
        for li in range(len(road.lanes)):
            ld = road.lanes[li]._lane_dict
            pld = prev_roads[-1].lanes[li]._lane_dict
            if ld["type"] != pld["type"]:
                return False
            if ld["speed_limit_mph"] != pld["speed_limit_mph"]:
                return False
            if ld["interpolating"] != pld["interpolating"]:
                return False
        return True

    def _create_roads_and_lanes(self, feat_splits: _FeatureSplits):
        for feat_id, splits in feat_splits.items():
            composite_roads = []
            split_inds = splits.sorted_keys
            assert len(split_inds) >= 2
            last_valid = split_inds[-1] - 1
            for s in range(len(split_inds) - 1):
                split_ind = split_inds[s]
                linked_split = splits[split_ind]
                assert (
                    linked_split.next_split
                    and linked_split.split.index < linked_split.next_split.split.index
                )
                if linked_split.split.index >= last_valid:
                    # XXX:  disallows 1-point polyline lanes, which do exist in about 15% of scenarios.
                    # To allow these requires changes that will cascade though.
                    # Practically this means that incoming_lanes ids will sometimes not be found.
                    continue
                if linked_split.used:
                    continue
                road_lanes = []
                seen = set()
                rt_structural, rt_junction = self._add_right_lanes(
                    linked_split, road_lanes, feat_splits, seen
                )
                lft_structural, lft_junction = self._add_left_lanes(
                    linked_split, road_lanes, feat_splits
                )
                structural = rt_structural or lft_structural
                junction = rt_junction or lft_junction
                road = self._create_road_from_lanes(road_lanes, junction)
                if self._no_composites:
                    continue
                if (
                    structural
                    or junction
                    or not WaymoMap._can_merge_roads(road, composite_roads)
                ):
                    if len(composite_roads) > 1:
                        self._create_composite(composite_roads)
                    composite_roads = []
                composite_roads.append(road)
            if len(composite_roads) > 1:
                self._create_composite(composite_roads)

    def _waymo_pb_to_dict(self, waymo_lane_feats) -> Dict[str, Any]:
        # we can't mutate the waymo protobuf objects, nor do they have a __dict__,
        # so we just keep the fields we're going to use...
        attribs = [
            "type",
            "interpolating",
            "entry_lanes",
            "exit_lanes",
            "speed_limit_mph",
            "left_boundaries",
            "right_boundaries",
            "left_neighbors",
            "right_neighbors",
        ]
        return {attr: getattr(waymo_lane_feats, attr) for attr in attribs}

    def _load_from_scenario(self, waymo_scenario):
        start = time.time()

        # cache feature info about lanes
        self._feat_dicts: Dict[int, Dict[str, Any]] = {}
        self._polyline_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        for map_feature in waymo_scenario.map_features:
            key = map_feature.WhichOneof("feature_data")
            if key is None:
                continue
            feat_id = map_feature.id
            map_feats = getattr(map_feature, key)
            self._waymo_features[feat_id] = map_feats
            if key != "lane":
                continue
            self._polyline_cache[feat_id] = WaymoMap._polyline_dists(map_feats.polyline)
            self._feat_dicts[feat_id] = self._waymo_pb_to_dict(map_feats)

            if len(self._polyline_cache[feat_id][0]) < 2:
                raise WaymoDatasetError(
                    f"[{self._waymo_scenario_id}] Feature {feat_id} only has a single point in its polyline, which is not currently supported by SMARTS."
                )

        # use original lane polylines for geometry
        for feat_id, lane_dict in self._feat_dicts.items():
            lane_dict["_normals"] = self._calculate_normals(feat_id)
            left_widths, right_widths = self._raycast_boundaries(lane_dict, feat_id)
            max_width = max(
                left_widths[0], left_widths[-1], right_widths[0], right_widths[-1]
            )
            if max_width < 0.5:
                max_width = WaymoMap.DEFAULT_LANE_WIDTH / 2
            max_width = min(max_width, WaymoMap.DEFAULT_LANE_WIDTH / 2)
            lane_dict["lane_width"] = max_width * 2

        feat_splits = self._find_splits()
        self._link_splits(feat_splits)
        self._create_roads_and_lanes(feat_splits)

        # don't need these anymore
        self._polyline_cache = None
        self._feat_dicts = None

        # find intersecting lanes
        self._compute_lane_intersections(composites=False)
        self._compute_lane_intersections(composites=True)

        # associate map features with surfaces
        for feat_id, map_feat_pb in self._waymo_features.items():
            if not isinstance(map_feat_pb, (StopSign, Crosswalk, SpeedBump)):
                continue
            feature_id = f"feature_{feat_id}"
            feature = WaymoMap.Feature(self, feature_id, map_feat_pb)
            self._features[feature_id] = feature
            if feature.type == RoadMap.FeatureType.STOP_SIGN:
                pos = self._map_pt_to_point(map_feat_pb.position)
                for lane, _ in self.nearest_lanes(pos):
                    if lane._feature_id in map_feat_pb.lane:
                        lane._features[feature_id] = feature
            else:
                # TODO:  use self.nearest_surface() (NYI) to find nearest
                # surfaces (lanes, roads, etc.) and add crosswalks and speed bumps
                # to their features.
                pass
        # also associate *fixed-location* traffic signals with lanes here
        # but handle the dynamic signals and states themselves elsewhere...
        lane_signals = {
            (ls.lane, self._map_pt_to_point(ls.stop_point))
            for ds in waymo_scenario.dynamic_map_states
            for ls in ds.lane_states
        }
        # remove non-fixed-location signals...
        static_lane_signals = dict()
        non_fixed = set()
        for lane_signal, stop_point in lane_signals:
            sp = static_lane_signals.setdefault(lane_signal, stop_point)
            if sp.x != stop_point.x and sp.y != stop_point.y:
                non_fixed.add(lane_signal)
        static_lane_signals = dict(
            filter(lambda item: item[0] not in non_fixed, static_lane_signals.items())
        )
        lane_sig_count = defaultdict(int)
        for lane_signal, stop_point in static_lane_signals.items():
            sp = self._map_pt_to_point(stop_point)
            for lane, _ in self.nearest_lanes(sp):
                if lane._feature_id == lane_signal:
                    lane_sig_count[lane_signal] += 1
                    feature_id = f"signal_{lane_signal}_{lane_sig_count[lane_signal]}"
                    feature = WaymoMap.Feature(self, feature_id, (stop_point, lane))
                    self._features[feature_id] = feature
                    lane._features[feature_id] = feature

        end = time.time()
        elapsed = round((end - start) * 1000.0, 3)
        self._log.info(f"Loading Waymo map took: {elapsed} ms")

    @staticmethod
    def parse_source_to_scenario(source: str):
        """Read the dataset file and get the specified scenario"""
        dataset_path = source.split("#")[0]
        scenario_id = source.split("#")[1]

        # Reset cache if this is a new TFRecord file
        if not WaymoMap._tfrecord_path or WaymoMap._tfrecord_path != dataset_path:
            WaymoMap._tfrecord_path = dataset_path
            WaymoMap._tfrecord_generator = read_tfrecord_file(dataset_path)
            WaymoMap._scenario_cache = dict()

        parsed_scenario = WaymoMap._scenario_cache.get(scenario_id)
        if parsed_scenario:
            return parsed_scenario

        while True:
            record = next(WaymoMap._tfrecord_generator, None)
            if not record:
                raise ValueError(
                    f"Dataset file does not contain scenario with id: {scenario_id}"
                )
            parsed_scenario = scenario_pb2.Scenario()
            parsed_scenario.ParseFromString(bytearray(record))
            WaymoMap._scenario_cache[parsed_scenario.scenario_id] = parsed_scenario
            if parsed_scenario.scenario_id == scenario_id:
                return parsed_scenario

    @classmethod
    def from_spec(cls, map_spec: MapSpec):
        """Generate a road network from the given specification."""
        if len(map_spec.source.split("#")) != 2:
            return None
        waymo_scenario = cls.parse_source_to_scenario(map_spec.source)
        assert waymo_scenario
        return cls(map_spec, waymo_scenario)

    @property
    def source(self) -> str:
        return self._map_spec.source

    @property
    def has_overpasses(self) -> bool:
        return self._has_overpasses

    @property
    def dynamic_features(self) -> List[RoadMap.Feature]:
        return [f for f in self._features.values() if f.is_dynamic]

    @staticmethod
    def _spec_lane_width(map_spec: MapSpec) -> float:
        return (
            map_spec.default_lane_width
            if map_spec.default_lane_width is not None
            else WaymoMap.DEFAULT_LANE_WIDTH
        )

    def is_same_map(self, map_spec: MapSpec) -> bool:
        waymo_scenario = WaymoMap.parse_source_to_scenario(map_spec.source)
        return (
            waymo_scenario.scenario_id == self._waymo_scenario_id
            and map_spec.lanepoint_spacing == self._map_spec.lanepoint_spacing
            and (
                map_spec.default_lane_width == self._map_spec.default_lane_width
                or WaymoMap._spec_lane_width(map_spec)
                == WaymoMap._spec_lane_width(self._map_spec)
            )
            and map_spec.shift_to_origin == self._map_spec.shift_to_origin
        )

    @cached_property
    def bounding_box(self) -> Optional[BoundingBox]:
        x_mins, y_mins, x_maxs, y_maxs = [], [], [], []
        for road_id in self._roads:
            road = self._roads[road_id]
            x_mins.append(road._bbox.min_pt.x)
            y_mins.append(road._bbox.min_pt.y)
            x_maxs.append(road._bbox.max_pt.x)
            y_maxs.append(road._bbox.max_pt.y)

        return BoundingBox(
            min_pt=Point(x=min(x_mins), y=min(y_mins)),
            max_pt=Point(x=max(x_maxs), y=max(y_maxs)),
        )

    @property
    def scale_factor(self) -> float:
        return 1.0  # TODO

    def to_glb(self, glb_dir):
        """Build a glb file for camera rendering and envision."""
        glb = self._make_glb_from_polys()
        glb.write_glb(Path(glb_dir) / "map.glb")

    def _make_glb_from_polys(self):
        scene = trimesh.Scene()
        polygons = []
        for lane_id in self._lanes:
            lane = self._lanes[lane_id]
            metadata = {
                "road_id": lane.road.road_id,
                "lane_id": lane_id,
                "lane_index": lane.index,
            }
            polygons.append((lane.shape(), metadata))

        meshes = generate_meshes_from_polygons(polygons)

        # Attach additional information for rendering as metadata in the map glb
        # <2D-BOUNDING_BOX>: four floats separated by ',' (<FLOAT>,<FLOAT>,<FLOAT>,<FLOAT>),
        # which describe x-minimum, y-minimum, x-maximum, and y-maximum
        metadata = {
            "bounding_box": (
                self.bounding_box.min_pt.x,
                self.bounding_box.min_pt.y,
                self.bounding_box.max_pt.x,
                self.bounding_box.max_pt.y,
            )
        }

        # lane markings information
        lane_dividers = self._compute_traffic_dividers()
        metadata["lane_dividers"] = lane_dividers

        for mesh in meshes:
            mesh.visual = trimesh.visual.TextureVisuals(
                material=trimesh.visual.material.PBRMaterial()
            )

            road_id = mesh.metadata["road_id"]
            lane_id = mesh.metadata.get("lane_id")
            name = f"{road_id}"
            if lane_id is not None:
                name += f"-{lane_id}"
            scene.add_geometry(mesh, name, extras=mesh.metadata)
        return _GLBData(gltf.export_glb(scene, extras=metadata, include_normals=True))

    def _compute_traffic_dividers(self):
        lane_dividers = []  # divider between lanes with same traffic direction
        for road_id in self._roads:
            road = self._roads[road_id]
            if not road.is_junction:
                for lane in road.lanes:
                    left_border_vertices_len = int((len(lane._lane_polygon) - 1) / 2)
                    left_side = lane._lane_polygon[:left_border_vertices_len]
                    lane_to_left, _ = lane.lane_to_left
                    if lane.index != len(road.lanes) - 1 and lane_to_left is not None:
                        if lane.is_drivable and lane_to_left.is_drivable:
                            lane_dividers.append(left_side)

        return lane_dividers

    class Surface(RoadMapWithCaches.Surface):
        """Surface representation for Waymo maps"""

        def __init__(self, surface_id: str, road_map):
            self._surface_id = surface_id
            self._map = road_map
            self._features: Dict[str, RoadMapWithCaches.Feature] = dict()

        @property
        def surface_id(self) -> str:
            return self._surface_id

        @property
        def is_drivable(self) -> bool:
            # XXX: this may be over-riden below
            return True

        @property
        def features(self) -> List[RoadMap.Feature]:
            return list(self._features.values())

        def features_near(self, pose: Pose, radius: float) -> List[RoadMap.Feature]:
            pt = pose.point
            return [
                feat
                for feat in self._features.values()
                if radius >= feat.min_dist_from(pt)
            ]

    def surface_by_id(self, surface_id: str) -> RoadMap.Surface:
        return self._surfaces.get(surface_id)

    class Lane(RoadMapWithCaches.Lane, Surface):
        """Lane representation for Waymo maps"""

        def __init__(self, road_map, lane_id: str, lane_dict: Dict[str, Any]):
            super().__init__(lane_id, road_map)
            self._feature_id = lane_dict["_feature_id"]
            self._road = None  # set when lane is added to a Road
            self._index = None  # set when lane is added to a Road
            self._lane_dict = lane_dict
            self._lane_pts = np.array(lane_dict["polyline"])
            self._centerline_pts = [Point(*p) for p in lane_dict["polyline"]]
            self._n_pts = len(self._lane_pts)
            self._lane_width = lane_dict["lane_width"]
            self._speed_limit = (
                lane_dict.get("speed_limit_mph", WaymoMap.DEFAULT_LANE_SPEED / 0.44704)
                * 0.44704
            )
            self._is_composite = bool(lane_dict.get("sublanes", None))
            self._length = sum(
                np.linalg.norm(self._lane_pts[i + 1][:2] - self._lane_pts[i][:2])
                for i in range(len(self._lane_pts) - 1)
            )
            self._drivable = lane_dict["type"] != LaneCenter.LaneType.TYPE_BIKE_LANE
            self._type = lane_dict["type"]

            self._lane_polygon = None
            self._create_polygon(lane_dict)
            if self._map._no_composites:
                del lane_dict["_normals"]
            x_coordinates, y_coordinates = zip(*self._lane_polygon)
            self._bbox = BoundingBox(
                min_pt=Point(x=min(x_coordinates), y=min(y_coordinates)),
                max_pt=Point(x=max(x_coordinates), y=max(y_coordinates)),
            )

        def __hash__(self) -> int:
            return hash(self.lane_id) + hash(self._map)

        def _create_polygon(self, lane_dict: Dict[str, Any]):
            new_left_pts = [None] * self._n_pts
            new_right_pts = [None] * self._n_pts
            for i in range(self._n_pts):
                p = self._lane_pts[i][:2]
                n = lane_dict["_normals"][i]
                w = self._lane_width / 2.0
                new_left_pts[i] = p + (w * n)
                new_right_pts[i] = p + (-1.0 * w * n)

            xs, ys = [], []
            for p in new_left_pts + new_right_pts[::-1] + [new_left_pts[0]]:
                if p is not None:
                    xs.append(p[0])
                    ys.append(p[1])
            self._lane_polygon = list(zip(xs, ys))

        @property
        def lane_id(self) -> str:
            return self._lane_id

        @property
        def road(self) -> RoadMap.Road:
            return self._road

        @property
        def in_junction(self) -> bool:
            return self._road.is_junction

        @property
        def index(self) -> int:
            return self._index

        @property
        def length(self) -> float:
            return self._length

        @property
        def is_drivable(self) -> bool:
            return self._drivable

        @property
        def center_polyline(self) -> List[Point]:
            return self._centerline_pts

        @property
        def composite_lane(self) -> RoadMapWithCaches.Lane:
            composite_id = self._lane_dict.get("composite")
            if composite_id:
                return self._map.lane_by_id(composite_id)
            return self

        @property
        def is_composite(self) -> bool:
            return self._is_composite

        @lru_cache(maxsize=4)
        def shape(
            self, buffer_width: float = 0.0, default_width: Optional[float] = None
        ) -> Polygon:
            if buffer_width == 0.0:
                return Polygon(self._lane_polygon)
            new_width = self._lane_width + buffer_width
            if new_width > 0:
                return buffered_shape(self._centerline_pts, new_width)
            return Polygon(self._lane_polygon)

        @cached_property
        def incoming_lanes(self) -> List[RoadMapWithCaches.Lane]:
            return [
                self._map.lane_by_id(il) for il in self._lane_dict["incoming_lane_ids"]
            ]

        @cached_property
        def outgoing_lanes(self) -> List[RoadMapWithCaches.Lane]:
            return [
                self._map.lane_by_id(ol) for ol in self._lane_dict["outgoing_lane_ids"]
            ]

        @cached_property
        def foes(self) -> List[RoadMapWithCaches.Lane]:
            result = {self._map.lane_by_id(ix) for ix in self._intersections}
            result |= {
                incoming
                for outgoing in self.outgoing_lanes
                for incoming in outgoing.incoming_lanes
                if incoming != self
            }
            return list(result)

        @property
        def entry_surfaces(self) -> List[RoadMap.Surface]:
            # TODO?  can a non-lane connect into a lane?
            return self.incoming_lanes

        @property
        def exit_surfaces(self) -> List[RoadMap.Surface]:
            # TODO?  can a lane exit to a non-lane?
            return self.outgoing_lanes

        @cached_property
        def lanes_in_same_direction(self) -> List[RoadMapWithCaches.Lane]:
            return [l for l in self.road.lanes if l != self]

        def _check_boundaries(self, split: "WaymoMap._Split", side: str) -> bool:
            neighbor = self._map._waymo_features[split.feat_id]
            for nb in getattr(neighbor, f"{side}_neighbors", []):
                for bd in nb.boundaries:
                    if (
                        bd.boundary_type
                        >= RoadLine.RoadLineType.TYPE_SOLID_DOUBLE_YELLOW
                    ):
                        return False
            return True

        def _adj_lane_info(self, adj_lane_info):
            if len(adj_lane_info) == 1:
                return adj_lane_info[0]
            min_fdelt = None
            for li in adj_lane_info:
                fdelt = abs(self._feature_id - li.feat_id)
                if not min_fdelt or fdelt < min_fdelt:
                    min_fdelt = fdelt
                    lane_info = li
            return lane_info

        def _get_side_lane(
            self, side: str
        ) -> Tuple[Optional[RoadMapWithCaches.Lane], bool]:
            li = self._lane_dict.get(f"lane_to_{side}_info")
            if not li:
                return None, True
            if isinstance(li, str):
                return self._map.lane_by_id(li), True
            li = self._adj_lane_info(li)
            same_dir = self._check_boundaries(li, "right" if side == "left" else "left")
            side_lane_id = WaymoMap._lane_id(li.feat_id, li.index)
            return self._map.lane_by_id(side_lane_id), same_dir

        @cached_property
        def lane_to_left(self) -> Tuple[Optional[RoadMapWithCaches.Lane], bool]:
            return self._get_side_lane("left")

        @cached_property
        def lane_to_right(self) -> Tuple[Optional[RoadMapWithCaches.Lane], bool]:
            return self._get_side_lane("right")

        @property
        def speed_limit(self) -> Optional[float]:
            return self._speed_limit

        def width_at_offset(self, lane_point_s: float) -> Tuple[float, float]:
            return self._lane_width, 1.0

        @lru_cache(maxsize=8)
        def center_at_point(self, point: Point) -> Point:
            return super().center_at_point(point)

        @lru_cache(maxsize=8)
        def center_pose_at_point(self, point: Point) -> Pose:
            return super().center_pose_at_point(point)

        @lru_cache(maxsize=1024)
        def curvature_radius_at_offset(
            self, offset: float, lookahead: int = 5
        ) -> float:
            return super().curvature_radius_at_offset(offset, lookahead)

        @lru_cache(maxsize=16)
        def oncoming_lanes_at_offset(
            self, offset: float
        ) -> List[RoadMapWithCaches.Lane]:
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

        @lru_cache(maxsize=8)
        def contains_point(self, point: Point) -> bool:
            assert type(point) == Point
            if (
                self._bbox.min_pt.x <= point[0] <= self._bbox.max_pt.x
                and self._bbox.min_pt.y <= point[1] <= self._bbox.max_pt.y
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
        ) -> Set[Tuple[RoadMapWithCaches.Lane, float]]:
            return super().project_along(start_offset, distance)

        def waypoint_paths_for_pose(
            self, pose: Pose, lookahead: int, route: RoadMap.Route = None
        ) -> List[List[Waypoint]]:
            if not self.is_drivable:
                return []
            road_ids = [road.road_id for road in route.roads] if route else None
            return self._waypoint_paths_at(pose.point, lookahead, road_ids)

        def waypoint_paths_at_offset(
            self, offset: float, lookahead: int = 30, route: RoadMap.Route = None
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

    class Road(RoadMapWithCaches.Road, Surface):
        """This is akin to a 'road segment' in real life.
        Many of these might correspond to a single named road in reality."""

        def __init__(
            self,
            road_map,
            road_lanes: Sequence[RoadMapWithCaches.Lane],
            is_junction: bool,
        ):
            self._composite = None
            self._is_composite = False
            self._is_junction = is_junction
            self._road_id = "waymo_road"

            self._drivable = False
            self._road_type = -1
            self._length = 0
            x_mins, y_mins, x_maxs, y_maxs = [], [], [], []
            for ind, lane in enumerate(road_lanes):
                self._road_id += f"-{lane.lane_id}"
                lane._road = self
                lane._index = ind
                self._length += lane.length
                x_mins.append(lane._bbox.min_pt.x)
                y_mins.append(lane._bbox.min_pt.y)
                x_maxs.append(lane._bbox.max_pt.x)
                y_maxs.append(lane._bbox.max_pt.y)
                if self._road_type == -1:
                    self._road_type = lane._type
                elif lane._type != self._road_type:
                    self._road_type = LaneCenter.LaneType.TYPE_UNDEFINED
                if lane.is_drivable:
                    self._drivable = True
                if lane.is_composite:
                    # TAI: do we need to keep track of sub roads?
                    self._is_composite = True

            self._length /= len(road_lanes)
            self._bbox = BoundingBox(
                min_pt=Point(x=min(x_mins), y=min(y_mins)),
                max_pt=Point(x=max(x_maxs), y=max(y_maxs)),
            )

            self._lanes = road_lanes
            self._compute_edge_shapes()

            super().__init__(self._road_id, road_map)

        def __hash__(self) -> int:
            return hash(self.road_id) ^ hash(self._map)

        @property
        def road_id(self) -> str:
            return self._road_id

        @property
        def type(self) -> int:
            return self._road_type

        @cached_property
        def type_as_str(self) -> str:
            road_type = self.type
            if road_type == 0:
                return "undefined"
            elif road_type == 1:
                return "freeway"
            elif road_type == 2:
                return "surface street"
            elif road_type == 3:
                return "bike lane"
            return "undefined"

        @property
        def is_drivable(self) -> bool:
            return self._drivable

        @property
        def composite_road(self) -> RoadMap.Road:
            return self._composite or self

        @property
        def is_composite(self) -> bool:
            return self._is_composite

        @cached_property
        def is_junction(self) -> bool:
            # XXX: Waymo does not indicate whether a road is in junction or not, but we can *sometimes* tell.
            if self._is_junction:
                return True
            for lane in self._lanes:
                if lane.foes or len(lane.incoming_lanes) > 1:
                    self._is_junction = True
                    return True
            return False

        @property
        def length(self) -> float:
            # Note: the more curved the road, the more the lane lengths diverge.
            return self._length

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

        def _compute_edge_shapes(self):
            leftmost_lane = self.lane_at_index(len(self._lanes) - 1)
            rightmost_lane = self.lane_at_index(0)

            rightmost_lane_buffered_polygon = rightmost_lane._lane_polygon
            leftmost_lane_buffered_polygon = leftmost_lane._lane_polygon

            # Right edge
            rightmost_edge_vertices_len = int(
                0.5 * (len(rightmost_lane_buffered_polygon) - 1)
            )
            self._rightmost_edge_shape = rightmost_lane_buffered_polygon[
                rightmost_edge_vertices_len : len(rightmost_lane_buffered_polygon) - 1
            ]

            # Left edge
            leftmost_edge_vertices_len = int(
                0.5 * (len(leftmost_lane_buffered_polygon) - 1)
            )
            self._leftmost_edge_shape = leftmost_lane_buffered_polygon[
                :leftmost_edge_vertices_len
            ]

        @lru_cache(maxsize=4)
        def shape(
            self, buffer_width: float = 0.0, default_width: Optional[float] = None
        ) -> Polygon:
            # TODO:  use buffer_width
            return Polygon(
                (
                    self._leftmost_edge_shape
                    + self._rightmost_edge_shape
                    + [self._leftmost_edge_shape[0]]
                )
            )

        @property
        def parallel_roads(self) -> List[RoadMap.Road]:
            return []

        @property
        def lanes(self) -> Sequence[RoadMapWithCaches.Lane]:
            return self._lanes

        def lane_at_index(self, index: int) -> RoadMapWithCaches.Lane:
            return self._lanes[index]

    def road_by_id(self, road_id: str) -> RoadMap.Road:
        road = self._roads.get(road_id)
        # XXX: If this asserts, it's probably because this map contains single-point polyline lanes, which we don't yet handle.
        assert road, f"WaymoMap got request for unknown road_id: '{road_id}'"
        return road

    def lane_by_id(self, lane_id: str) -> RoadMapWithCaches.Lane:
        # note: all lanes were cached already by _load()
        lane = self._lanes.get(lane_id)
        # XXX: If this asserts, it's probably because this map contains single-point polyline lanes, which we don't yet handle.
        assert lane, f"WaymoMap got request for unknown lane_id: '{lane_id}'"
        return lane

    @lru_cache(maxsize=4)
    def dynamic_features_near(
        self, point: Point, radius: float
    ) -> List[Tuple[RoadMap.Feature, float]]:
        return super().dynamic_features_near(point, radius)

    @cached_property
    def _simple_lanes(self) -> List[RoadMapWithCaches.Lane]:
        return [lane for lane in self._lanes.values() if not lane.is_composite]

    def _build_lane_r_tree(self):
        result = rtree.index.Index()
        result.interleaved = True
        # only index simple lanes, as composite lanes can
        # always be gotten from a simple lane, and we don't
        # want more ambiguity in our spatial queries.
        for idx, lane in enumerate(self._simple_lanes):
            bounding_box = (
                lane._bbox.min_pt.x,
                lane._bbox.min_pt.y,
                lane._bbox.max_pt.x,
                lane._bbox.max_pt.y,
            )
            result.add(idx, bounding_box)
        return result

    def _get_neighboring_lanes(
        self, x: float, y: float, r: float = 0.1
    ) -> List[Tuple[RoadMapWithCaches.Lane, float]]:
        neighboring_lanes = []
        if self._lane_rtree is None:
            self._lane_rtree = self._build_lane_r_tree()

        simple_lanes = self._simple_lanes
        spt = SPoint(x, y)
        for i in self._lane_rtree.intersection((x - r, y - r, x + r, y + r)):
            lane = simple_lanes[i]
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
            radius = max(10, 2 * self._default_lane_width)
        candidate_lanes = self._get_neighboring_lanes(point[0], point[1], r=radius)
        candidate_lanes.sort(key=lambda lane_dist_tup: lane_dist_tup[1])
        return candidate_lanes

    def nearest_lane(
        self,
        point: Point,
        radius: Optional[float] = None,
        include_junctions: bool = False,
    ) -> Optional[RoadMapWithCaches.Lane]:
        nearest_lanes = self.nearest_lanes(point, radius, include_junctions)
        for lane, dist in nearest_lanes:
            if lane.contains_point(point):
                # Since Waymo has lanes of varying width, a point can be closer to a lane it does not lie in
                # when compared to the lane it does if it is closer to the outer lane's central line,
                # than the lane it lies in.
                return lane
        return nearest_lanes[0][0] if nearest_lanes else None

    @lru_cache(maxsize=16)
    def road_with_point(self, point: Point) -> RoadMap.Road:
        radius = max(5, 2 * self._default_lane_width)
        for nl, dist in self.nearest_lanes(point, radius):
            if nl.contains_point(point):
                return nl.road
        return None

    class Feature(RoadMap.Feature):
        """Feature representation for Waymo maps"""

        def __init__(self, road_map, feature_id: str, feat_proto):
            self._map = road_map
            self._feature_id = feature_id
            self._feat_proto = feat_proto
            self._type = self._proto_type_to_type(feat_proto)

        @staticmethod
        def _proto_type_to_type(feat_proto) -> int:
            if isinstance(feat_proto, Crosswalk):
                return RoadMap.FeatureType.CROSSWALK
            if isinstance(feat_proto, SpeedBump):
                return RoadMap.FeatureType.SPEED_BUMP
            if isinstance(feat_proto, StopSign):
                return RoadMap.FeatureType.STOP_SIGN
            if isinstance(feat_proto, tuple):
                return RoadMap.FeatureType.FIXED_LOC_SIGNAL
            return RoadMap.FeatureType.UNKNOWN

        @property
        def feature_id(self) -> str:
            return self._feature_id

        @property
        def type(self) -> RoadMap.FeatureType:
            return self._type

        @property
        def type_as_str(self) -> str:
            return self._type.name

        @property
        def geometry(self) -> List[Point]:
            if isinstance(self._feat_proto, tuple):
                return [self._feat_proto[0]]
            point = getattr(self._feat_proto, "position", None)
            if point:
                return [self._map._map_pt_to_point(point)]
            polygon = getattr(self._feat_proto, "polygon", None)
            if polygon:
                return [self._map._map_pt_to_point(pt) for pt in polygon]
            return []

        @cached_property
        def type_specific_info(self) -> Optional[Any]:
            if self._type == RoadMap.FeatureType.FIXED_LOC_SIGNAL:
                return self._feat_proto[1]
            return None

        def min_dist_from(self, point: Point) -> float:
            pt = point.as_np_array
            return min(
                np.linalg.norm(geo_pt.as_np_array - pt) for geo_pt in self.geometry
            )

    def feature_by_id(self, feature_id: str) -> RoadMap.Feature:
        return self._features.get(feature_id)

    class Route(RouteWithCache):
        """Describes a route between Waymo roads."""

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

        def _add_road(self, road: RoadMap.Road):
            self._length += road.length
            self._roads.append(road)

        @cached_property
        def geometry(self) -> Sequence[Sequence[Tuple[float, float]]]:
            return [list(road.shape().exterior.coords) for road in self.roads]

    @staticmethod
    def _shortest_route(start: RoadMap.Road, end: RoadMap.Road) -> List[RoadMap.Road]:
        queue = [(start.length, start.road_id, start)]
        came_from = dict()
        came_from[start] = None
        cost_so_far = dict()
        cost_so_far[start] = start.length
        current = None

        # Dijkstra’s Algorithm
        while queue:
            (_, _, current) = heapq.heappop(queue)
            current: RoadMap.Road
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
            path.append(current)
            current = came_from[current]
        path.append(start)
        path.reverse()
        return path

    def generate_routes(
        self,
        start_road: RoadMap.Road,
        end_road: RoadMap.Road,
        via: Optional[Sequence[RoadMap.Road]] = None,
        max_to_gen: int = 1,
    ) -> List[RoadMap.Route]:
        assert max_to_gen == 1, "multiple route generation not yet supported for Waymo"
        new_route = WaymoMap.Route(self)
        result = [new_route]

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
            sub_route = WaymoMap._shortest_route(cur_road, next_road) or []
            if len(sub_route) < 2:
                self._log.warning(
                    f"Unable to find valid path between {(cur_road.road_id, next_road.road_id)}."
                )
                return result
            # The sub route includes the boundary roads (cur_road, next_road).
            # We clip the latter to prevent duplicates
            route_roads.extend(sub_route[:-1])

        for road in route_roads:
            new_route._add_road(road)
        return result

    def random_route(
        self,
        max_route_len: int = 10,
        starting_road: Optional[RoadMap.Road] = None,
        only_drivable: bool = True,
    ) -> RoadMap.Route:
        assert not starting_road or not only_drivable or starting_road.is_drivable
        route = WaymoMap.Route(self)
        next_roads = [starting_road] if starting_road else list(self._roads.values())
        if only_drivable:
            next_roads = [r for r in next_roads if r.is_drivable]
        while next_roads and len(route.roads) < max_route_len:
            cur_road = random.choice(next_roads)
            route._add_road(cur_road)
            next_roads = list(cur_road.outgoing_roads)
        return route

    def empty_route(self) -> RoadMap.Route:
        return WaymoMap.Route(self)

    def route_from_road_ids(self, road_ids: Sequence[str]) -> RoadMap.Route:
        return WaymoMap.Route.from_road_ids(self, road_ids)

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
            self._starts[llp.lp.lane.index] = paths

        def query(
            self,
            lookahead: int,
            point: Point,
            filter_road_ids: tuple,
            llp,
        ) -> Optional[List[List[Waypoint]]]:
            """Attempt to find previously cached waypoints"""
            if self._match(lookahead, point, filter_road_ids):
                hit = self._starts.get(llp.lp.lane.index, None)
                if hit:
                    # consider just returning all of them (not slicing)?
                    return [path[: (lookahead + 1)] for path in hit]
                return None

    def waypoint_paths(
        self,
        pose: Pose,
        lookahead: int,
        within_radius: float = 5,
        route: RoadMap.Route = None,
    ) -> List[List[Waypoint]]:
        road_ids = []
        if route and route.roads:
            road_ids = [road.road_id for road in route.roads]
        if road_ids:
            return self._waypoint_paths_along_route(pose.point, lookahead, road_ids)
        closest_lps = self._lanepoints.closest_lanepoints(
            [pose], within_radius=within_radius
        )
        closest_lane = closest_lps[0].lane
        waypoint_paths = []
        for lane in closest_lane.road.lanes:
            waypoint_paths += lane._waypoint_paths_at(pose.point, lookahead)
        return sorted(waypoint_paths, key=lambda p: p[0].lane_index)

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
        Waypoints that may not be evenly spaced (due to lane change) but start at point."""

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
            WaymoMap._equally_spaced_path(
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
