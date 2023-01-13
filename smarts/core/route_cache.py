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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import logging
import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Set, Tuple

from cached_property import cached_property

from .road_map import RoadMap

# cache_keys shouldn't be exposed/used outside of this Route
_RouteKey = int


@dataclass
class _LaneContinuation:
    """Struct containing information about the future of a Lane along a Route."""

    dist_to_end: float = 0.0
    dist_to_junction: float = math.inf
    next_junction: RoadMap.Lane = None
    dist_to_road: Dict[RoadMap.Road, float] = field(default_factory=dict)


_route_sub_lengths: Dict[
    _RouteKey, Dict[RoadMap.Route.RouteLane, _LaneContinuation]
] = dict()


class RouteWithCache(RoadMap.Route):
    """A cache for commonly-needed but expensive-to-compute information about RoadMap.Routes."""

    def __init__(self, road_map: RoadMap):
        self._map = road_map
        self._logger = logging.getLogger(self.__class__.__name__)

    def __hash__(self) -> int:
        return self._cache_key

    def __eq__(self, other) -> bool:
        return self.__class__ == other.__class__ and hash(self) == hash(other)

    def _add_road(self, road: RoadMap.Road):
        raise NotImplementedError()

    @cached_property
    def road_ids(self) -> List[str]:
        """Retruns a list of the road_ids for the Roads in this Route."""
        return [road.road_id for road in self.roads]

    @classmethod
    def from_road_ids(
        cls, road_map: RoadMap, road_ids: Sequence[str]
    ) -> "RouteWithCache":
        """Factory to generate a new RouteWithCache from a sequence of road ids."""
        route = cls(road_map)
        for road_id in road_ids:
            road = road_map.road_by_id(road_id)
            assert road, f"cannot add unknown road {road_id} to route"
            route._add_road(road)
        return route

    @cached_property
    def _cache_key(self) -> _RouteKey:
        return hash(tuple(road.road_id for road in self.roads)) ^ hash(self._map)

    @property
    def is_cached(self) -> bool:
        """Returns True if information about this Route has been cached."""
        return self._cache_key in _route_sub_lengths

    def remove_from_cache(self):
        """Remove informationa about this Route from the cache."""
        if self.is_cached:
            del _route_sub_lengths[self._cache_key]

    # TAI: could pre-cache curvatures here too (like waypoints) ?
    def add_to_cache(self):
        """Add informationa about this Route to the cache if not already there."""
        if self.is_cached:
            return

        cache_key = self._cache_key
        _route_sub_lengths[cache_key] = dict()

        def _backprop_length(
            bplane: RoadMap.Lane,
            length: float,
            rind: int,
            junction: bool,
            final_lane: RoadMap.Lane,
        ):
            assert rind >= 0
            rind -= 1
            for il in bplane.incoming_lanes:
                rl = RoadMap.Route.RouteLane(il, rind)
                il_cont = _route_sub_lengths[cache_key].get(rl)
                if il_cont is not None:
                    if junction:
                        if il.in_junction:
                            junction = False
                        else:
                            il_cont.dist_to_junction = il_cont.dist_to_end
                            il_cont.next_junction = final_lane
                    il_cont.dist_to_road[final_lane.road] = il_cont.dist_to_end
                    il_cont.dist_to_end += length
                    _backprop_length(il, length, rind, junction, final_lane)

        road = None
        for r_ind, road in enumerate(self.roads):
            for lane in road.lanes:
                # r_ind is required to correctly handle routes with sub-cycles
                rl = RoadMap.Route.RouteLane(lane, r_ind)
                assert rl not in _route_sub_lengths[cache_key]
                _backprop_length(lane, lane.length, r_ind, lane.in_junction, lane)
                lc = _LaneContinuation(lane.length)
                if lane.in_junction:
                    lc.next_junction = lane
                    lc.dist_to_junction = 0.0
                _route_sub_lengths[cache_key][rl] = lc

        if not road:
            return

        # give lanes that would form a loop an advantage...
        first_road = self.roads[0]
        for lane in road.lanes:
            rl = RoadMap.Route.RouteLane(lane, r_ind)
            for og in lane.outgoing_lanes:
                if og.road == first_road:
                    _route_sub_lengths[cache_key][rl].dist_to_end += 1

    def _find_along(
        self, rpt: RoadMap.Route.RoutePoint, radius: float = 30.0
    ) -> Optional[RoadMap.Route.RouteLane]:
        for cand_lane, _ in self._map.nearest_lanes(
            rpt.pt, radius, include_junctions=True
        ):
            try:
                rind = self._roads.index(cand_lane.road)
                if rind >= 0 and (rpt.road_index is None or rpt.road_index == rind):
                    return RoadMap.Route.RouteLane(cand_lane, rind)
            except ValueError:
                pass
        self._logger.warning("Unable to find road on route near point %s", rpt)
        return None

    @lru_cache(maxsize=8)
    def distance_between(
        self, start: RoadMap.Route.RoutePoint, end: RoadMap.Route.RoutePoint
    ) -> Optional[float]:
        rt_ln = self._find_along(start)
        if not rt_ln:
            return None
        start_lane = rt_ln.lane
        sind = rt_ln.road_index
        start_road = start_lane.road

        rt_ln = self._find_along(end)
        if not rt_ln:
            return None
        end_lane = rt_ln.lane
        eind = rt_ln.road_index
        end_road = end_lane.road

        d = 0.0
        start_offset = start_lane.offset_along_lane(start.pt)
        end_offset = end_lane.offset_along_lane(end.pt)
        if start_road == end_road and sind == eind:
            return end_offset - start_offset
        negate = False
        if sind > eind:
            start_lane = end_lane
            start_road, end_road = end_road, start_road
            start_offset, end_offset = end_offset, start_offset
            negate = True
        d = end_offset + start_lane.length - start_offset
        for rind, road in enumerate(self._roads):
            if rind >= eind:
                break
            if rind <= sind:
                continue
            d += road.length
        return -d if negate else d

    @lru_cache(maxsize=8)
    def project_along(
        self, start: RoadMap.Route.RoutePoint, distance: float
    ) -> Optional[Set[Tuple[RoadMap.Lane, float]]]:
        rt_ln = self._find_along(start)
        if not rt_ln:
            return None
        start_lane = rt_ln.lane
        sind = rt_ln.road_index

        orig_offset = start_lane.offset_along_lane(start.pt)
        for rind, road in enumerate(self._roads):
            if rind < sind:
                continue
            start_offset = 0 if rind != sind else orig_offset
            if distance > road.length - start_offset:
                distance -= road.length - start_offset
                continue
            return {(lane, distance) for lane in road.lanes}
        return set()

    def distance_from(
        self,
        cur_lane: RoadMap.Route.RouteLane,
        route_road: Optional[RoadMap.Road] = None,
    ) -> Optional[float]:
        self.add_to_cache()
        lc = _route_sub_lengths[self._cache_key].get(cur_lane)
        if not lc:
            return None
        if route_road:
            return lc.dist_to_road.get(route_road)
        return lc.dist_to_end

    def next_junction(
        self, cur_lane: RoadMap.Route.RouteLane, offset: float
    ) -> Tuple[Optional[RoadMap.Lane], float]:
        self.add_to_cache()
        lc = _route_sub_lengths[self._cache_key].get(cur_lane)
        if lc:
            dist = lc.dist_to_junction
            if lc.dist_to_junction > 0:
                dist -= offset
                assert dist >= 0
            return lc.next_junction, dist
        return None, math.inf
