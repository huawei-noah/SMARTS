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
import random
import re
import weakref
import xml.etree.ElementTree as XET
from bisect import bisect_left, bisect_right, insort
from collections import defaultdict, deque
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

import numpy as np
from cached_property import cached_property
from shapely.affinity import rotate as shapely_rotate
from shapely.geometry import Polygon
from shapely.geometry import box as shapely_box

from .actor import ActorRole, ActorState
from .controllers import ActionSpaceType
from .coordinates import Dimensions, Heading, Point, Pose, RefLinePoint
from .provider import Provider, ProviderManager, ProviderRecoveryFlags, ProviderState
from .road_map import RoadMap
from .route_cache import RouteWithCache
from .scenario import Scenario
from .signals import SignalLightState, SignalState
from .traffic_provider import TrafficProvider
from .utils.kinematics import (
    distance_covered,
    stopping_distance,
    stopping_time,
    time_to_cover,
)
from .utils.math import min_angles_difference_signed, radians_to_vec, vec_to_radians
from .vehicle import VEHICLE_CONFIGS, VehicleState


def _safe_division(n: float, d: float, default=math.inf):
    """This method uses a short circuit form where `and` converts right side to true|false (as 1|0) in which cases are:
    True and # == #
    False and NaN == False
    """
    return d and n / d or default


class LocalTrafficProvider(TrafficProvider):
    """A LocalTrafficProvider simulates multiple traffic actors on a generic RoadMap."""

    def __init__(self):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._sim = None
        self._scenario = None
        self.road_map: RoadMap = None
        self._flows: Dict[str, Dict[str, Any]] = dict()
        self._my_actors: Dict[str, _TrafficActor] = dict()
        self._other_actors: Dict[
            str, Tuple[ActorState, Optional[RoadMap.Route]]
        ] = dict()
        self._reserved_areas: Dict[str, Polygon] = dict()
        self._actors_created: int = 0
        self._lane_bumpers_cache: Dict[
            RoadMap.Lane, List[Tuple[float, VehicleState, int]]
        ] = dict()
        self._offsets_cache: Dict[str, Dict[str, float]] = dict()
        # start with the default recovery flags...
        self._recovery_flags = super().recovery_flags

    @property
    def recovery_flags(self) -> ProviderRecoveryFlags:
        return self._recovery_flags

    @recovery_flags.setter
    def recovery_flags(self, flags: ProviderRecoveryFlags):
        self._recovery_flags = flags

    def set_manager(self, manager: ProviderManager):
        self._sim = weakref.ref(manager)

    @property
    def actions(self) -> Set[ActionSpaceType]:
        return set()

    def manages_actor(self, actor_id: str) -> bool:
        return actor_id in self._my_actors

    def _load_traffic_flows(self, traffic_spec: str):
        vtypes = {}
        routes = {}
        root = XET.parse(traffic_spec).getroot()
        assert root.tag == "routes"
        for child in root:
            if child.tag == "vType":
                vid = child.attrib["id"]
                vtypes[vid] = child.attrib
            elif child.tag == "route":
                rid = child.attrib["id"]
                routes[rid] = child.attrib["edges"]
            elif child.tag == "flow":
                flow = child.attrib
                vtype = vtypes.get(flow["type"])
                assert vtype, f"undefined vehicle type {flow['type']} used in flow"
                flow["vtype"] = vtype
                route = routes.get(flow["route"])
                assert route, f"undefined route {flow['route']} used in flow"
                route = self.road_map.route_from_road_ids(route.split())
                route.add_to_cache()
                flow["route"] = route
                flow["begin"] = float(flow["begin"])
                flow["end"] = float(flow["end"])
                if "vehsPerHour" in flow:
                    freq = float(flow["vehsPerHour"])
                    if freq <= 0:
                        logging.warning(
                            f"vehPerHour value is {freq}<=0 vehicles may not be emitted!!!"
                        )
                        freq = 0
                    flow["emit_period"] = _safe_division(3600.0, freq)
                elif "period" in flow:
                    period = float(flow["period"])
                    assert period > 0.0
                    flow["emit_period"] = period
                elif "probability" in flow:
                    emit_prob = float(flow["probability"])
                    assert 0.0 <= emit_prob <= 1.0
                    flow["emit_prob"] = emit_prob
                else:
                    assert (
                        False
                    ), "either 'vehsPerHour' or 'probability' must be specified for Flow emission"
                flow_id = str(flow["id"])
                flow["endless"] = "endless" in flow_id
                self._flows[flow_id] = flow

    def _check_actor_bbox(self, actor: "_TrafficActor") -> bool:
        actor_bbox = actor.bbox(True)
        for reserved_area in self._reserved_areas.values():
            if reserved_area.intersects(actor_bbox):
                return False
        for my_actor in self._my_actors.values():
            if actor != my_actor and my_actor.bbox().intersects(actor_bbox):
                return False
        for other, _ in self._other_actors.values():
            if isinstance(other, VehicleState) and other.bbox.intersects(actor_bbox):
                return False
        return True

    def _add_actor_in_flow(self, flow: Dict[str, Any]) -> bool:
        new_actor = _TrafficActor.from_flow(flow, self)
        if not self._check_actor_bbox(new_actor):
            return False
        self._my_actors[new_actor.actor_id] = new_actor
        self._logger.info(f"traffic actor {new_actor.actor_id} entered simulation")
        return True

    def _add_actors_for_time(self, sim_time: float, dt: float = 1.0):
        for flow in self._flows.values():
            if not flow["begin"] <= sim_time < flow["end"]:
                continue
            try_add = False
            last_added = flow.get("last_added")
            emit_prob = flow.get("emit_prob")
            emit_period = flow.get("emit_period")
            if emit_period is not None:
                try_add = last_added is None or sim_time - last_added >= emit_period
            elif emit_prob is not None:
                try_add = random.random() <= emit_prob * dt
            if try_add and self._add_actor_in_flow(flow):
                flow["last_added"] = sim_time

    @property
    def _my_actor_states(self) -> List[VehicleState]:
        return [actor.state for actor in self._my_actors.values()]

    @property
    def _other_vehicle_states(self) -> List[VehicleState]:
        return [
            other
            for other, _ in self._other_actors.values()
            if isinstance(other, VehicleState)
        ]

    @property
    def _all_states(self) -> List[VehicleState]:
        return self._my_actor_states + self._other_vehicle_states

    @property
    def _provider_state(self) -> ProviderState:
        return ProviderState(actors=self._my_actor_states)

    def setup(self, scenario: Scenario) -> ProviderState:
        assert self._sim() is not None
        self._scenario = scenario
        self.road_map = scenario.road_map
        traffic_specs = [
            ts for ts in self._scenario.traffic_specs if ts.endswith(".smarts.xml")
        ]
        assert len(traffic_specs) <= 1
        if traffic_specs:
            self._load_traffic_flows(traffic_specs[0])
        # TAI: is there any point if not?
        self._add_actors_for_time(0.0)
        return self._provider_state

    def _create_actor_caches(self):
        self._offsets_cache = dict()
        self._lane_bumpers_cache = dict()
        for ovs in self._all_states:
            center = ovs.pose.point
            length = ovs.dimensions.length
            hhx, hhy = radians_to_vec(ovs.pose.heading) * (0.5 * length)
            back = Point(center.x - hhx, center.y - hhy)
            front = Point(center.x + hhx, center.y + hhy)
            back_lane = self.road_map.nearest_lane(back, radius=length)
            front_lane = self.road_map.nearest_lane(front, radius=length)
            if back_lane:
                back_offset = back_lane.offset_along_lane(back)
                lbc = self._lane_bumpers_cache.setdefault(back_lane, [])
                insort(lbc, (back_offset, ovs, 1))
            if front_lane:
                front_offset = front_lane.offset_along_lane(front)
                lbc = self._lane_bumpers_cache.setdefault(front_lane, [])
                insort(lbc, (front_offset, ovs, 2))
            if front_lane and back_lane != front_lane:
                # it's changing lanes, don't misjudge the target lane...
                fake_back_offset = front_lane.offset_along_lane(back)
                insort(self._lane_bumpers_cache[front_lane], (fake_back_offset, ovs, 0))

    def _cached_lane_offset(self, vs: VehicleState, lane: RoadMap.Lane):
        lane_offsets = self._offsets_cache.setdefault(vs.actor_id, dict())
        return lane_offsets.setdefault(
            lane.lane_id, lane.offset_along_lane(vs.pose.point)
        )

    def _relinquish_actor(self, actor_state: ActorState):
        sim = self._sim()
        assert sim
        sim.provider_relinquishing_actor(self, actor_state)
        self._logger.debug(
            f"{actor_state} is no longer managed by local traffic provider"
        )
        if actor_state.actor_id in self._my_actors:
            del self._my_actors[actor_state.actor_id]

    def step(self, actions, dt: float, elapsed_sim_time: float) -> ProviderState:
        sim = self._sim()
        assert sim
        self._add_actors_for_time(elapsed_sim_time, dt)
        for other in self._other_vehicle_states:
            if other.actor_id in self._reserved_areas:
                del self._reserved_areas[other.actor_id]

        # precompute nearest lanes and offsets for all vehicles and cache
        # (this prevents having to do it O(ovs^2) times)
        self._create_actor_caches()

        # Do state update in two passes so that we don't use next states in the
        # computations for actors encountered later in the iterator.
        for actor in self._my_actors.values():
            actor.compute_next_state(dt)

        dones = set()
        losts = set()
        removed = set()
        remap_ids: Dict[str, str] = dict()
        for actor_id, actor in self._my_actors.items():
            actor.step(dt)
            if actor.finished_route:
                dones.add(actor.actor_id)
            elif actor.off_route:
                losts.add(actor)
            elif actor.teleporting:
                # pybullet doesn't like it when a vehicle jumps from one side of the map to another,
                # so we need to give teleporting vehicles a new id and thus a new chassis.
                actor.bump_id()
                remap_ids[actor_id] = actor.actor_id
        for actor in losts - removed:
            removed.add(actor.actor_id)
            self._relinquish_actor(actor.state)
        for actor_id in dones - removed:
            actor = self._my_actors.get(actor_id)
            if actor:
                sim.provider_removing_actor(self, actor_id)
            # The following is not really necessary due to the above calling teardown(),
            # but it doesn't hurt...
            if actor_id in self._my_actors:
                del self._my_actors[actor_id]
        for orig_id, new_id in remap_ids.items():
            self._my_actors[new_id] = self._my_actors[orig_id]
            del self._my_actors[orig_id]

        return self._provider_state

    def sync(self, provider_state: ProviderState):
        missing = self._my_actors.keys() - {
            psv.actor_id for psv in provider_state.actors
        }
        for left in missing:
            self._logger.warning(
                f"locally provided actor '{left}' disappeared from simulation"
            )
            del self._my_actors[left]
        hijacked = self._my_actors.keys() & {
            psv.actor_id
            for psv in provider_state.actors
            if psv.source != self.source_str
        }
        for jack in hijacked:
            self.stop_managing(jack)
        self._other_actors = dict()
        for os in provider_state.actors:
            my_actor = self._my_actors.get(os.actor_id)
            if my_actor:
                assert os.source == self.source_str
                # here we override our state with the "consensus" state...
                # (Note: this is different from what Sumo does;
                # we're allowing for true "harmonization" if necessary.
                # "You may say that I'm a dreamer, but I'm not the only one,
                # I hope that one day you'll join us, and the world will
                # be as one." ;)
                my_actor.state = os
            else:
                assert os.source != self.source_str
                self._other_actors[os.actor_id] = (os, None)

    def reset(self):
        # Unify interfaces with other providers
        pass

    def teardown(self):
        self._my_actors = dict()
        self._other_actors = dict()
        self._reserved_areas = dict()

    def destroy(self):
        pass

    def stop_managing(self, actor_id: str):
        # called when agent hijacks this vehicle
        self._logger.debug(f"{actor_id} is removed from local traffic management")
        assert (
            actor_id in self._my_actors
        ), f"stop_managing() called for non-tracked vehicle id '{actor_id}'"
        del self._my_actors[actor_id]

    def reserve_traffic_location_for_vehicle(
        self,
        vehicle_id: str,
        reserved_location: Polygon,
    ):
        self._reserved_areas[vehicle_id] = reserved_location

    def vehicle_collided(self, vehicle_id: str):
        traffic_actor = self._my_actors.get(vehicle_id)
        if not traffic_actor:
            # guess we already removed it for some other reason (off route?)
            return
        # TAI:  consider relinquishing / removing the vehicle? Like:
        # self._relinquish_actor(traffic_actor.state)
        # If collidee(s) include(s) an EgoAgent, it will likely be # marked "done" and things will end anyway.
        # (But this is not guaranteed depending on the done criteria that were set.)
        # Probably the most realistic thing we can do is leave the vehicle sitting in the road, blocking traffic!
        # (... and then add a "rubber-neck mode" for all nearby vehicles?! ;)
        # Let's do that for now, but we should also consider just removing the vehicle.
        # traffic_actor.stay_put()

    def update_route_for_vehicle(self, vehicle_id: str, new_route: RoadMap.Route):
        traffic_actor = self._my_actors.get(vehicle_id)
        if traffic_actor:
            traffic_actor.update_route(new_route)
            return
        other = self._other_actors.get(vehicle_id)
        if other:
            self._other_actors[vehicle_id] = (other[0], new_route)
            return
        assert False, f"unknown vehicle_id: {vehicle_id}"

    def route_for_vehicle(self, vehicle_id: str) -> Optional[RoadMap.Route]:
        traffic_actor = self._my_actors.get(vehicle_id)
        if traffic_actor:
            return traffic_actor.route
        other = self._other_actors.get(vehicle_id)
        if other:
            _, oroute = other
            return oroute if oroute else None
        assert False, f"unknown vehicle_id: {vehicle_id}"

    def vehicle_dest_road(self, vehicle_id: str) -> Optional[str]:
        route = self.route_for_vehicle(vehicle_id)
        return route.road_ids[-1]

    def can_accept_actor(self, state: ActorState) -> bool:
        # We don't accept vehicles that aren't on the road
        # (those should currently be removed from the simultation).
        return (
            isinstance(state, VehicleState)
            and (state.role == ActorRole.Social or state.role == ActorRole.Unknown)
            and self.road_map.nearest_lane(state.pose.point) is not None
        )

    def add_actor(
        self, provider_actor: ActorState, from_provider: Optional[Provider] = None
    ):
        assert isinstance(provider_actor, VehicleState)
        route = None
        if from_provider and isinstance(from_provider, TrafficProvider):
            route = from_provider.route_for_vehicle(provider_actor.actor_id)
            assert not route or isinstance(route, RouteWithCache)
        provider_actor.source = self.source_str
        provider_actor.role = ActorRole.Social
        xfrd_actor = _TrafficActor.from_state(provider_actor, self, route)
        self._my_actors[xfrd_actor.actor_id] = xfrd_actor
        if xfrd_actor.actor_id in self._other_actors:
            del self._other_actors[xfrd_actor.actor_id]
        self._logger.info(
            f"traffic actor {xfrd_actor.actor_id} transferred to {self.source_str}."
        )

    def _signal_state_means_stop(
        self, lane: RoadMap.Lane, signal_feat: RoadMap.Feature
    ) -> bool:
        feat_state = self._other_actors.get(signal_feat.feature_id)
        if not feat_state:
            return False
        feat_state = feat_state[0]
        assert isinstance(feat_state, SignalState)
        return not (feat_state.state & SignalLightState.GO)

    def _stopped_at_features(self, actor_id: str) -> List[str]:
        actor = self._my_actors.get(actor_id)
        if actor:
            return actor.stopped_at_features
        return []


class _TrafficActor:
    """Simulates a vehicle managed by the LocalTrafficProvider."""

    def __init__(self, flow: Dict[str, Any], owner: LocalTrafficProvider):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._owner = weakref.ref(owner)

        self._state = None
        self._flow: Dict[str, Any] = flow
        self._vtype: Dict[str, Any] = flow["vtype"]
        self._route_ind: int = 0
        self._done_with_route: bool = False
        self._off_route: bool = False
        self._route: RouteWithCache = flow["route"]
        self._stranded: bool = False
        self._teleporting: bool = False

        self._lane = None
        self._offset = 0
        self._lane_speed: Dict[int, Tuple[float, float]] = dict()
        self._lane_windows: Dict[int, _TrafficActor._LaneWindow] = dict()
        self._lane_win: _TrafficActor._LaneWindow = None
        self._target_lane_win: _TrafficActor._LaneWindow = None
        self._target_speed: float = 15.0  # arbitrary reasonable default
        self._dest_lane = None
        self._dest_offset = None
        self._wrong_way: bool = False

        # The default values here all attempt to match those in sstudio.types,
        # which in turn attempt to match Sumo's defaults.
        self._min_space_cush = float(self._vtype.get("minGap", 2.5))
        speed_factor = float(self._vtype.get("speedFactor", 1.0))
        speed_dev = float(self._vtype.get("speedDev", 0.1))
        self._speed_factor = random.gauss(speed_factor, speed_dev)
        if self._speed_factor <= 0:
            self._speed_factor = 0.1  # arbitrary minimum speed is 10% of speed limit
        self._imperfection = float(self._vtype.get("sigma", 0.5))
        self._max_accel = float(self._vtype.get("accel", 2.6))
        assert self._max_accel >= 0.0
        self._max_decel = float(self._vtype.get("decel", 4.5))
        assert self._max_decel >= 0.0

        self._cutting_into = None
        self._cutting_in = False
        self._in_front_after_cutin_secs = 0
        self._cutin_hold_secs = float(self._vtype.get("lcHoldPeriod", 10.0))
        self._forward_after_added = 0
        self._after_added_hold_secs = self._cutin_hold_secs
        self._target_cutin_gap = 2.5 * self._min_space_cush
        self._aggressiveness = float(self._vtype.get("lcAssertive", 1.0))
        if self._aggressiveness <= 0:
            self._logger.warning(
                "non-positive value {self._aggressiveness} for 'assertive' lane-changing parameter will be ignored"
            )
            self._aggressiveness = 1.0
        self._cutin_prob = float(self._vtype.get("lcCutinProb", 0.0))
        if not 0.0 <= self._cutin_prob <= 1.0:
            self._logger.warning(
                "illegal probability {self._cutin_prob} for 'cutin_prob' lane-changing parameter will be ignored"
            )
            self._cutin_prob = 0.0
        self._dogmatic = self._vtype.get("lcDogmatic", "False") == "True"
        self._cutin_slow_down = float(self._vtype.get("lcSlowDownAfter", 1.0))
        if self._cutin_slow_down < 0:
            self._cutin_slow_down = 0
        self._multi_lane_cutin = self._vtype.get("lcMultiLaneCutin", "False") == "True"
        self._yield_to_agents = self._vtype.get("jmYieldToAgents", "normal")
        self._wait_to_restart = float(self._vtype.get("jmWaitToRestart", 0.0))
        self._stopped_at_feat = dict()
        self._waiting_at_feat = defaultdict(float)

        self._max_angular_velocity = 26  # arbitrary, based on limited testing
        self._prev_angular_err = None

        self._bumper_wins_front = _TrafficActor._RelWindow(5)
        self._bumper_wins_back = _TrafficActor._RelWindow(5)

        owner._actors_created += 1

    @classmethod
    def from_flow(cls, flow: Dict[str, Any], owner: LocalTrafficProvider):
        """Factory to construct a _TrafficActor object from a flow dictionary."""
        vclass = flow["vtype"]["vClass"]
        dimensions = VEHICLE_CONFIGS[vclass].dimensions
        vehicle_type = vclass if vclass != "passenger" else "car"

        new_actor = cls(flow, owner)
        new_actor._lane, new_actor._offset = new_actor._resolve_flow_pos(
            flow, "depart", dimensions
        )
        position = new_actor._lane.from_lane_coord(RefLinePoint(s=new_actor._offset))
        heading = vec_to_radians(
            new_actor._lane.vector_at_offset(new_actor._offset)[:2]
        )
        init_speed = new_actor._resolve_flow_speed(flow)
        endless = "-endless" if flow.get("endless", False) else ""
        vehicle_id = f"{new_actor._vtype['id']}{endless}-{owner._actors_created}"

        new_actor._state = VehicleState(
            actor_id=vehicle_id,
            actor_type=vehicle_type,
            source=owner.source_str,
            role=ActorRole.Social,
            pose=Pose.from_center(position, Heading(heading)),
            dimensions=dimensions,
            vehicle_config_type=vclass,
            speed=init_speed,
            linear_acceleration=np.array((0.0, 0.0, 0.0)),
        )
        new_actor._dest_lane, new_actor._dest_offset = new_actor._resolve_flow_pos(
            flow, "arrival", dimensions
        )
        return new_actor

    @classmethod
    def from_state(
        cls,
        state: VehicleState,
        owner: LocalTrafficProvider,
        route: Optional[RouteWithCache],
    ):
        """Factory to construct a _TrafficActor object from an existing VehiclState object."""
        cur_lane = owner.road_map.nearest_lane(state.pose.point)
        assert (
            cur_lane
        ), f"LocalTrafficProvider accepted a vehicle that's off road?  pos={state.pose.point}"
        if not route or not route.roads:
            # initialize with a random route, which will probably be replaced
            route = owner.road_map.random_route(starting_road=cur_lane.road)
        route.add_to_cache()
        flow = dict()
        flow["vtype"] = dict()
        flow["route"] = route
        flow["arrivalLane"] = f"{cur_lane.index}"  # XXX: assumption!
        flow["arrivalPos"] = "max"
        flow["departLane"] = f"{cur_lane.index}"  # XXX: assumption!
        flow["departPos"] = "0"
        flow["endless"] = "endless" in state.actor_id
        # use default values for everything else in flow dict(s)...
        new_actor = _TrafficActor(flow, owner)
        new_actor.state = state
        new_actor._lane = cur_lane
        new_actor._offset = cur_lane.offset_along_lane(state.pose.point)
        new_actor._dest_lane, new_actor._dest_offset = new_actor._resolve_flow_pos(
            flow, "arrival", state.dimensions
        )
        return new_actor

    def _resolve_flow_pos(
        self, flow: Dict[str, Any], depart_arrival: str, dimensions: Dimensions
    ) -> Tuple[RoadMap.Lane, float]:
        base_err = (
            f"scenario traffic specifies flow with invalid route {depart_arrival} point"
        )
        road = (
            self._route.roads[0]
            if depart_arrival == "depart"
            else self._route.roads[-1]
        )
        assert road, "invalid route"
        lane_ind = max(0, min(int(flow[f"{depart_arrival}Lane"]), len(road.lanes) - 1))
        lane = road.lanes[lane_ind]
        offset = flow[f"{depart_arrival}Pos"]
        if offset == "max":
            offset = max(lane.length - 0.5 * dimensions.length, 0)
        elif offset == "random":
            offset = random.random() * lane.length
        elif 0 <= float(offset) <= lane.length:
            offset = float(offset)
        else:
            raise Exception(
                f"{base_err}:  starting offset {offset} invalid for road_id '{road.road_id}'."
            )
        target_pt = lane.from_lane_coord(RefLinePoint(s=offset))
        offset = lane.offset_along_lane(target_pt)
        return (lane, offset)

    def _resolve_flow_speed(self, flow: Dict[str, Any]) -> float:
        depart_speed = flow.get("departSpeed", 0.0)
        # maxSpeed's default attempts to match the one in sstudio.types,
        # which in turn attempt to match Sumo's default.
        max_speed = float(self._vtype.get("maxSpeed", 55.5))
        if depart_speed == "random":
            return random.random() * max_speed
        elif depart_speed == "max":
            return min(max_speed, self._lane.speed_limit)
        elif depart_speed == "speedLimit":
            if self._lane.speed_limit is not None:
                return self._lane.speed_limit
            else:
                raise Exception(
                    f"scenario specifies departSpeed='speed_limit' but no speed limit defined for lane '{self._lane.lane_id}'."
                )
        departSpeed = float(depart_speed)
        assert departSpeed >= 0
        return departSpeed

    @property
    def state(self) -> VehicleState:
        """Returns the current VehicleState for this actor."""
        return self._state

    @state.setter
    def state(self, state: VehicleState):
        """Sets the current VehicleState for this actor."""
        self._state = state
        self.bbox.cache_clear()

    @property
    def actor_id(self) -> str:
        """A unique id identifying this actor."""
        return self._state.actor_id

    def bump_id(self):
        """Changes the id of a teleporting vehicle."""
        mm = re.match(r"([^_]+)(_(\d+))?$", self._state.actor_id)
        if mm:
            ver = int(mm.group(3)) if mm.group(3) else 0
            self._state.actor_id = f"{mm.group(1)}_{ver + 1}"

    @property
    def route(self) -> RoadMap.Route:
        """The route this actor will attempt to take."""
        return self._route

    def update_route(self, route: RouteWithCache):
        """Update the route (sequence of road_ids) this actor will attempt to take.
        A unique route_key is provided for referencing the route cache in he owner provider."""
        self._route = route
        self._route.add_to_cache()
        self._dest_lane, self._dest_offset = self._resolve_flow_pos(
            self._flow, "arrival", self._state.dimensions
        )
        self._route_ind = 0

    def stay_put(self):
        """Tells this actor to stop acting and remain where it is indefinitely."""
        if not self._stranded:
            self._logger.info(f"{self.actor_id} stranded")
            self._stranded = True
            self._state.speed = 0
            self._state.linear_acceleration = None

    @property
    def finished_route(self) -> bool:
        """Returns True iff this vehicle has reached the end of its route."""
        return self._done_with_route

    @property
    def off_route(self) -> bool:
        """Returns True iff this vehicle has left its route before it got to the end."""
        return self._off_route

    @property
    def wrong_way(self) -> bool:
        """Returns True iff this vehicle is currently going the wrong way in its lane."""
        return self._wrong_way

    @property
    def teleporting(self) -> bool:
        """Returns True iff this vehicle is teleporting back to the beginning of its route on this step."""
        return self._teleporting

    @property
    def lane(self) -> RoadMap.Lane:
        """Returns the current Lane object."""
        return self._lane

    @property
    def road(self) -> RoadMap.Road:
        """Returns the current Road object."""
        return self._lane.road

    @property
    def offset_along_lane(self) -> float:
        """Returns the current offset along the current Lane object."""
        return self._offset

    @property
    def speed(self) -> float:
        """Returns the current speed."""
        return self._state.speed

    @property
    def acceleration(self) -> float:
        """Returns the current (linear) acceleration."""
        if self._state.linear_acceleration is None:
            return 0.0
        return np.linalg.norm(self._state.linear_acceleration)

    @lru_cache(maxsize=2)
    def bbox(self, cushion_length: bool = False) -> Polygon:
        """Returns a bounding box around the vehicle."""
        # note: lru_cache must be cleared whenever pose changes
        pos = self._state.pose.point
        dims = self._state.dimensions
        length_buffer = self._min_space_cush if cushion_length else 0
        half_len = 0.5 * dims.length + length_buffer
        poly = shapely_box(
            pos.x - 0.5 * dims.width,
            pos.y - half_len,
            pos.x + 0.5 * dims.width,
            pos.y + half_len,
        )
        return shapely_rotate(poly, self._state.pose.heading, use_radians=True)

    @property
    def stopped_at_features(self) -> List[str]:
        """If this vehicle is currently stopped at any features,
        returns their feature_ids, otherwise an empty list."""
        return list(self._stopped_at_feat.keys())

    class _LaneWindow:
        def __init__(
            self,
            lane: RoadMap.Lane,
            time_left: float,
            ttc: float,
            ttre: float,
            gap: float,
            lane_coord: RefLinePoint,
            agent_gap: Optional[float],
            ahead_id: Optional[str],
        ):
            self.lane = lane
            self.time_left = time_left
            self.adj_time_left = time_left  # could eventually be negative
            self.ttc = ttc
            self.ttre = ttre  # time until we'd get rear-ended
            self.gap = gap  # just the gap ahead (in meters)
            self.lane_coord = lane_coord
            self.agent_gap = agent_gap
            self.ahead_id = ahead_id

        @property
        def drive_time(self) -> float:
            """The amount of time this vehicle might drive in the
            target lane under present conditions."""
            return min(self.ttc, self.adj_time_left)

        @cached_property
        def width(self) -> float:
            """The width of this lane at its lane_coord."""
            return self.lane.width_at_offset(self.lane_coord.s)[0]

        @cached_property
        def radius(self) -> float:
            """The radius of curvature of this lane at its lane_coord."""
            # we round the offset in an attempt to reduce the unique hits on the LRU caches...
            rounded_offset = round(self.lane_coord.s)
            return self.lane.curvature_radius_at_offset(
                rounded_offset, lookahead=max(math.ceil(2 * self.width), 2)
            )

        @lru_cache(maxsize=4)
        def _angle_scale(self, to_index: int, theta: float = math.pi / 6) -> float:
            # we need to correct for not going straight across.
            # other things being equal, we target ~30 degrees (sin(30)=.5) on average.
            if abs(self.radius) > 1e5 or self.radius == 0:
                return _safe_division(1.0, math.sin(theta), 1e6)
            # here we correct for the local road curvature (which affects how far we must travel)...
            T = _safe_division(self.radius, self.width, 1e6)
            # XXX: This cannot be an assertion because it could happen for map related reasons.
            if abs(T) <= 1.0:
                logging.debug(
                    "abnormally high curvature?  radius=%s, width=%s at offset %s of lane %s",
                    self.radius,
                    self.width,
                    self.lane_coord.s,
                    self.lane.lane_id,
                )
            if to_index > self.lane.index:
                se = T * (T - 1)
                return math.sqrt(
                    2
                    * (
                        se
                        + 0.5
                        - se
                        * math.cos(
                            _safe_division(1, (math.tan(theta) * (T - 1)), default=0)
                        )
                    )
                )
            se = T * (T + 1)
            return math.sqrt(
                2
                * (
                    se
                    + 0.5
                    - se
                    * math.cos(
                        _safe_division(1, (math.tan(theta) * (T + 1)), default=0)
                    )
                )
            )

        def crossing_time_at_speed(
            self, to_index: int, speed: float, acc: float = 0.0
        ) -> float:
            """Returns how long it would take to cross from this lane to
            the lane indexed by to_index given our current speed and acceleration."""
            angle_scale = self._angle_scale(to_index)
            return time_to_cover(angle_scale * self.width, speed, acc)

        @lru_cache(maxsize=8)
        def exit_time(self, speed: float, to_index: int, acc: float = 0.0) -> float:
            """Returns how long it would take to drive into the to_index lane
            from this lane given our current speed and acceleration."""
            ct = self.crossing_time_at_speed(to_index, speed, acc)
            t = self.lane_coord.t
            pm = (-1 if to_index >= self.lane.index else 1) * np.sign(t)
            angle_scale = self._angle_scale(to_index)
            return 0.5 * ct + pm * time_to_cover(angle_scale * abs(t), speed, acc)

    def _find_vehicle_ahead_on_route(
        self, lane: RoadMap.Lane, dte: float, rind: int
    ) -> Tuple[float, Optional[VehicleState]]:
        owner = self._owner()
        assert owner
        nv_ahead_dist = math.inf
        nv_ahead_vs = None
        rind += 1
        for ogl in lane.outgoing_lanes:
            rt_oln = RoadMap.Route.RouteLane(ogl, rind)
            len_to_end = self._route.distance_from(rt_oln)
            if len_to_end is None:
                continue
            lbc = owner._lane_bumpers_cache.get(ogl)
            if lbc:
                fi = 0
                while fi < len(lbc):
                    ogl_offset, ogl_vs, _ = lbc[fi]
                    if ogl_vs.actor_id != self.actor_id:
                        break
                    fi += 1
                if fi == len(lbc):
                    continue
                ogl_dist = dte - (len_to_end - ogl_offset)
                if ogl_dist < nv_ahead_dist:
                    nv_ahead_dist = ogl_dist
                    nv_ahead_vs = ogl_vs
                continue
            ogl_dist, ogl_vs = self._find_vehicle_ahead_on_route(ogl, dte, rind)
            if ogl_dist < nv_ahead_dist:
                nv_ahead_dist = ogl_dist
                nv_ahead_vs = ogl_vs
        return nv_ahead_dist, nv_ahead_vs

    def _find_vehicle_ahead(
        self, lane: RoadMap.Lane, my_offset: float, search_start: float
    ) -> Tuple[float, Optional[VehicleState]]:
        owner = self._owner()
        assert owner
        lbc = owner._lane_bumpers_cache.get(lane)
        if lbc:
            lane_spot = bisect_right(lbc, (search_start, self._state, 3))
            # if we're at an angle to the lane, it's possible for the
            # first thing we hit to be our own entries in the bumpers cache,
            # which we need to skip.
            while lane_spot < len(lbc) and self.actor_id == lbc[lane_spot][1].actor_id:
                lane_spot += 1
            if lane_spot < len(lbc):
                lane_offset, nvs, _ = lbc[lane_spot]
                assert lane_offset >= search_start
                if lane_offset > my_offset:
                    return lane_offset - my_offset, nvs
                return 0, nvs
        rt_ln = RoadMap.Route.RouteLane(lane, self._route_ind)
        route_len = self._route.distance_from(rt_ln) or lane.length
        my_dist_to_end = route_len - my_offset
        return self._find_vehicle_ahead_on_route(lane, my_dist_to_end, self._route_ind)

    def _find_vehicle_behind(
        self, lane: RoadMap.Lane, my_offset: float, search_start: float
    ) -> Tuple[float, Optional[VehicleState]]:
        owner = self._owner()
        assert owner
        lbc = owner._lane_bumpers_cache.get(lane)
        if lbc:
            lane_spot = bisect_left(lbc, (search_start, self._state, -1))
            # if we're at an angle to the lane, it's possible for the
            # first thing we hit to be our own entries in the bumpers cache,
            # which we need to skip.
            while lane_spot > 0 and self.actor_id == lbc[lane_spot - 1][1].actor_id:
                lane_spot -= 1
            if lane_spot > 0:
                lane_offset, bv_vs, _ = lbc[lane_spot - 1]
                assert lane_offset <= search_start
                if lane_offset < my_offset:
                    return my_offset - lane_offset, bv_vs
                return 0, bv_vs

        def find_last(ll: RoadMap.Lane) -> Tuple[float, Optional[VehicleState]]:
            owner = self._owner()
            assert owner
            plbc = owner._lane_bumpers_cache.get(ll)
            if not plbc:
                return math.inf, None
            for bv_offset, bv_vs, _ in reversed(plbc):
                if bv_vs.actor_id != self.actor_id:
                    return bv_offset, bv_vs
            return math.inf, None

        # look back until split entry or vehicle found
        min_to_look = 100
        rind = self._route_ind - 1
        dist_looked = my_offset
        while len(lane.incoming_lanes) == 1 and rind >= 0:
            lane = lane.incoming_lanes[0]
            rind -= 1
            dist_looked += lane.length
            bv_offset, bv_vs = find_last(lane)
            if bv_vs:
                return dist_looked - bv_offset, bv_vs
        if rind == 0 or dist_looked > min_to_look:
            return math.inf, None

        # we hit a split without looking very far...
        # so go down each split branch until looked 100m back
        def _backtrack_to_len(ll, looked, rind) -> Tuple[float, Optional[VehicleState]]:
            if rind < 0 or looked >= min_to_look:
                return math.inf, None
            best_offset = math.inf
            best_vs = None
            for inl in ll.incoming_lanes:
                il_offset, il_vs = find_last(inl)
                if il_vs:
                    return looked + inl.length - il_offset, il_vs
                il_offset, il_vs = _backtrack_to_len(inl, looked + inl.length, rind - 1)
                if il_offset < best_offset:
                    best_offset = il_offset
                    best_vs = il_vs
            return best_offset, best_vs

        return _backtrack_to_len(lane, dist_looked, rind)

    def _compute_lane_window(self, lane: RoadMap.Lane):
        lane_coord = lane.to_lane_coord(self._state.pose.point)
        my_offset = lane_coord.s
        my_speed, my_acc = self._lane_speed[lane.index]
        rt_ln = RoadMap.Route.RouteLane(lane, self._route_ind)
        path_len = self._route.distance_from(rt_ln) or lane.length
        path_len -= my_offset
        lane_time_left = _safe_division(path_len, self.speed)

        half_len = 0.5 * self._state.dimensions.length
        front_bumper = my_offset + half_len
        back_bumper = my_offset - half_len

        ahead_dist, nv_vs = self._find_vehicle_ahead(lane, front_bumper, back_bumper)
        if nv_vs:
            ahead_dist -= self._min_space_cush
            ahead_dist = max(0, ahead_dist)
            speed_delta = my_speed - nv_vs.speed
            acc_delta = my_acc
            if nv_vs.linear_acceleration is not None:
                acc_delta -= np.linalg.norm(nv_vs.linear_acceleration)
            lane_ttc = max(time_to_cover(ahead_dist, speed_delta, acc_delta), 0)
        else:
            lane_ttc = math.inf

        behind_dist, bv_vs = self._find_vehicle_behind(lane, back_bumper, front_bumper)
        if bv_vs:
            behind_dist -= self._min_space_cush
            behind_dist = max(0, behind_dist)
            speed_delta = my_speed - bv_vs.speed
            acc_delta = my_acc
            if bv_vs.linear_acceleration is not None:
                acc_delta -= np.linalg.norm(bv_vs.linear_acceleration)
            lane_ttre = max(time_to_cover(behind_dist, -speed_delta, -acc_delta), 0)
        else:
            lane_ttre = math.inf

        self._lane_windows[lane.index] = _TrafficActor._LaneWindow(
            lane,
            lane_time_left,
            lane_ttc,
            lane_ttre,
            ahead_dist,
            lane_coord,
            behind_dist if bv_vs and bv_vs.role == ActorRole.EgoAgent else None,
            nv_vs.actor_id if nv_vs else None,
        )

    def _compute_lane_windows(self):
        self._lane_windows = dict()
        for lane in self.road.lanes:
            self._compute_lane_window(lane)
        for index, lw in self._lane_windows.items():
            lw.adj_time_left -= self._crossing_time_into(index)[0]

    def _crossing_time_into(self, target_idx: int) -> Tuple[float, bool]:
        my_idx = self._lane.index
        if my_idx == target_idx:
            return 0.0, True
        accel = self.acceleration
        max_speed = (
            self._lane_windows[target_idx].lane.speed_limit * self._speed_factor
        ) or 1e-13
        if self.speed < max_speed:
            # using my current acceleration is too conservative b/c I could
            # accelerate to a new speed as I change lanes (for example,
            # if moving from a stopped lane of traffic to a new one).
            bumped_accel = self._max_accel * (1.0 - self.speed / max_speed)
            accel = max(bumped_accel, accel, self._max_accel)
        min_idx = min(target_idx, my_idx + 1)
        max_idx = max(target_idx + 1, my_idx)
        cross_time = self._lane_windows[my_idx].exit_time(self.speed, target_idx, accel)
        for i in range(min_idx, max_idx):
            lw = self._lane_windows[i]
            lct = lw.crossing_time_at_speed(target_idx, self.speed, accel)
            if i == target_idx:
                lct *= 0.75
            cross_time += lct
        # note: we *could* be more clever and use cross_time for each lane separately
        # to try to thread our way through the gaps independently... nah.
        for i in range(min_idx, max_idx):
            lw = self._lane_windows[i]
            if min(lw.ttc, lw.time_left, lw.ttre) <= cross_time:
                return cross_time, False
        return cross_time, True

    def _should_cutin(self, lw: _LaneWindow) -> bool:
        target_ind = lw.lane.index
        if target_ind == self._lane.index:
            return False
        if not self._multi_lane_cutin and abs(target_ind - self._lane.index) > 1:
            return False
        if not self._dogmatic and lw.time_left < stopping_time(
            self.speed, self._max_decel
        ):
            return False
        min_gap = _safe_division(
            self._target_cutin_gap, self._aggressiveness, default=1e5
        )
        max_gap = self._target_cutin_gap + 2
        if min_gap < lw.agent_gap < max_gap and self._crossing_time_into(target_ind)[1]:
            return random.random() < self._cutin_prob
        return False

    def _pick_lane(self, dt: float):
        # first, survey the state of the road around me
        self._compute_lane_windows()
        my_idx = self._lane.index
        self._lane_win = self._lane_windows[my_idx]
        # Try to find the best among available lanes...
        best_lw = self._lane_windows[my_idx]

        # Default current lane then check right lanes then left lanes.
        lanes_to_right = list(range(0, my_idx))[::-1]
        lanes_to_left = list(
            range(min(my_idx + 1, len(self._lane_windows)), len(self._lane_windows))
        )
        checks = lanes_to_right + lanes_to_left
        # hold lane for some time if added recently
        if self._forward_after_added < self._after_added_hold_secs:
            self._forward_after_added += dt
            # skip checks
            checks = []

        ## TODO: Determine how blocked lane changes should be addressed
        ## Idea is to keep lane if blocked on right, slow down if blocked on left
        # if doing_cut_in and blocked_from_cut_in:
        #     # blocked on the right so pick a closer lane until cutin lane is available
        #     if blocked_on_right:
        #         # exclude blocked lane from lane checks
        #     # if blocked_on_left:
        #         # do nothing for now and wait

        for idx in checks:
            lw = self._lane_windows[idx]
            # skip lanes I can't drive in (e.g., bike lanes on waymo maps)
            if not lw.lane.is_drivable:
                continue
            # if I can't safely reach the lane, don't consider it
            change_time = 0
            if abs(idx - my_idx) > 1:
                change_time, can_cross = self._crossing_time_into(idx)
                if not can_cross:
                    continue
            min_time_cush = float(self._vtype.get("tau", 1.0))
            neighbour_lane_bias = (
                0.1 * change_time * (1 if abs(self._lane.index - idx) == 1 else 0)
            )
            will_rearend = lw.ttc + neighbour_lane_bias < min_time_cush
            # if my route destination is available, prefer that
            if (
                lw.lane == self._dest_lane
                and lw.lane_coord.s + lw.gap >= self._dest_offset
            ):
                # TAI: speed up or slow down as appropriate if _crossing_time_into() was False
                best_lw = lw
                if not will_rearend and not self._dogmatic:
                    break
            cut_in_is_real_lane = self._cutting_into and self._cutting_into.index < len(
                self._lane_windows
            )
            # if I'm in the process of changing lanes, continue (unless it's no longer safe)
            if (
                cut_in_is_real_lane
                and self._crossing_time_into(self._cutting_into.index)[1]
                and not will_rearend
            ):
                best_lw = self._lane_windows[self._cutting_into.index]
                if self._cutting_into != self._lane:
                    break
                # so I'm finally in the target cut-in lane, but now I gotta
                # stay there a bit to not appear indecisive
                self._in_front_after_cutin_secs += dt
                if self._in_front_after_cutin_secs < self._cutin_hold_secs:
                    break
            self._cutting_into = None
            self._cutting_in = False
            self._in_front_secs = 0
            # don't change lanes in junctions, except for the above reasons
            # (since it makes collision avoidance harder for everyone!)
            if lw.lane.in_junction:
                continue
            if change_time < lw.time_left:
                # also don't change lanes if we can't *finish* before entering a junction
                # (unless our lane is ending)
                rl = RoadMap.Route.RouteLane(lw.lane, self._route_ind)
                owner = self._owner()
                assert owner
                l_offset = owner._cached_lane_offset(self._state, lw.lane)
                _, nj_dist = self._route.next_junction(rl, l_offset)
                if change_time > time_to_cover(nj_dist, self.speed, self.acceleration):
                    continue
            # if there's an agent behind me, possibly cut in on it
            if lw.agent_gap and self._should_cutin(lw):
                best_lw = lw
                self._cutting_into = lw.lane
                self._cutting_in = True
                continue
            longer_drive_time = lw.drive_time > best_lw.drive_time
            equal_drive_time = lw.drive_time == best_lw.drive_time
            is_destination_lane = lw.lane == self._dest_lane
            highest_ttre = lw.ttre >= best_lw.ttre
            right_of_current_lw = idx < self._lane_win.lane.index
            # otherwise, keep track of the remaining options and eventually
            # pick the lane with the longest available driving time on my route
            # or, in the case of ties, the right-most lane (assuming I'm not
            # cutting anyone off to get there).

            if equal_drive_time and not will_rearend:
                if is_destination_lane and self._offset < self._dest_offset:
                    best_lw = lw
                if highest_ttre and right_of_current_lw:
                    best_lw = lw

            if longer_drive_time:
                best_lw = lw

            if will_rearend and lw.ttc > best_lw.ttc:
                best_lw = lw

        # keep track of the fact I'm changing lanes for next step
        # so I don't swerve back and forth indecesively
        if best_lw.lane != self._lane and not self._cutting_into:
            self._cutting_into = best_lw.lane
        self._target_lane_win = best_lw

    def _compute_lane_speeds(self):
        owner = self._owner()
        assert owner

        def _get_radius(lane: RoadMap.Lane) -> float:
            l_offset = owner._cached_lane_offset(self._state, lane)
            # we round the offset in an attempt to reduce the unique hits on the LRU caches...
            l_offset = round(l_offset)
            l_width, _ = lane.width_at_offset(l_offset)
            return lane.curvature_radius_at_offset(
                l_offset, lookahead=max(math.ceil(2 * l_width), 2)
            )

        self._lane_speed = dict()
        my_radius = _get_radius(self._lane)
        for l in self._lane.road.lanes:
            ratio = 1.0
            if l != self._lane and abs(my_radius) < 1e5:
                l_radius = _get_radius(l)
                if abs(l_radius) < 1e5:
                    ratio = _safe_division(l_radius, my_radius, default=0)
                    if ratio < 0:
                        ratio = 1.0
            self._lane_speed[l.index] = (ratio * self.speed, ratio * self.acceleration)

    def _slow_for_curves(self):
        # XXX:  this may be too expensive.  if so, we'll need to precompute curvy spots for routes
        lookahead = math.ceil(1 + math.log(self._target_speed))
        lookahead = max(1, lookahead)
        # we round the offset in an attempt to reduce the unique hits on the LRU caches...
        rounded_offset = round(self._offset)
        radius = self._lane.curvature_radius_at_offset(rounded_offset, lookahead)
        # pi/2 radian right turn == 6 m/s with radius 10.5 m
        # TODO:  also depends on vehicle type (traction, length, etc.)
        self._target_speed = min(abs(radius) * 0.5714, self._target_speed)

    class _RelWindow:
        @dataclass
        class _RelativeVehInfo:
            dist: float
            bearing: float
            heading: float
            dt: float

        def __init__(self, width: int = 5):
            self._width = width
            self._junction_foes: Dict[
                Tuple[str, bool], Deque[_TrafficActor._RelWindow._RelativeVehInfo]
            ] = dict()

        def add_to_win(
            self,
            veh_id: str,
            bumper: int,
            fv_pos: np.ndarray,
            my_pos: np.ndarray,
            my_heading: float,
            dt: float,
        ) -> Tuple[float, float]:
            """Add a relative observation of veh_id over the last dt secs."""
            bvec = fv_pos - my_pos
            fv_range = np.linalg.norm(bvec)
            rel_bearing = min_angles_difference_signed(vec_to_radians(bvec), my_heading)
            fv_win = self._junction_foes.setdefault(
                (veh_id, bumper), deque(maxlen=self._width)
            )
            fv_win.append(self._RelativeVehInfo(fv_range, rel_bearing, my_heading, dt))
            return rel_bearing, fv_range

        def predict_crash_in(self, veh_id: str, bumper: int) -> float:
            """Estimate if and when I might collide with veh_id using
            the constant bearing, decreasing range (CBDR) techique, but
            attempting to correct for non-linearities."""
            # CBDR is a good approximation at distance over time with
            # both vehicles behaving "steadily" (ideally, driving
            # straight at a constant speed).  It breaks down, thankfully
            # in preictable ways, if these assumptions are violated.
            #
            # Range Dependency:
            # As range gets small, bearing can fluctuate more, but will also
            # tend to start to change steadily when on a collision course.
            # To overcome this here, we just make our "CB" computation
            # depend on range.
            #
            # When range is on the order of car length(s), the following might eventually
            # help with evasive maneuvers:
            # - for side impacts to me, the absolute value of the rel_bearing from my front bumper
            #   will increase monotonically (and rapidly) just before impact, meanwhile the value
            #   from my back bumper will decrease analogously.
            # - for side impacts to them, the rel_bearing to their front bumper from mine will change
            #   signs just before impact, while abs(rel_bearing) to their back bumper will dramatically
            #   reduce.
            #
            # Heading Changes:
            # - If we turn, we need to consider how this will change rel_bearing from our perspective.
            #   Adding to the heading will increase apparent rel_bearing, but this instaneous change
            #   should not count against CBDR, so we correct for it here.
            # - If they turn, we don't need to do anything (CBDR as implemented here still works
            #   to predict collisions).
            #
            # (Relative) Acceleration:
            # - if there is a difference in our accelerations, rel_bearing for a collision course
            #   will change quadratically in a complicated way.  We don't try to mitigate this.
            window = self._junction_foes[(veh_id, bumper)]
            if len(window) <= 1:
                return math.inf
            prev_range, prev_bearing, prev_heading = None, None, None
            range_del = 0.0
            bearing_del = 0.0
            for rvi in window:
                if prev_range is not None:
                    range_del += (rvi.dist - prev_range) / rvi.dt
                if prev_bearing is not None:
                    bearing_del += (
                        min_angles_difference_signed(rvi.bearing, prev_bearing)
                        + min_angles_difference_signed(rvi.heading, prev_heading)
                    ) / rvi.dt
                prev_range = rvi.dist
                prev_bearing = rvi.bearing
                prev_heading = rvi.heading
            range_del /= len(window) - 1
            bearing_del /= len(window) - 1
            final_range = window[-1].dist

            # the exponent here was determined by trial and error
            if range_del < 0 and abs(bearing_del) < _safe_division(
                math.pi, final_range**1.4
            ):
                return _safe_division(-final_range, range_del)
            return math.inf

        def purge_unseen(self, seen: Set[str]):
            """Remove unseen vehicle IDs from this cache."""
            self._junction_foes = {
                (sv, bumper): self._junction_foes[(sv, bumper)]
                for sv in seen
                for bumper in (False, True)
                if (sv, bumper) in self._junction_foes
            }

    @staticmethod
    @lru_cache(maxsize=32)
    def _turn_angle(junction: RoadMap.Lane, approach_index: int) -> float:
        # TAI: consider moving this into RoadMap.Lane
        if junction.outgoing_lanes:
            next_lane = junction.outgoing_lanes[0]
            nlv = next_lane.vector_at_offset(0.5 * next_lane.length)
        else:
            nlv = junction.vector_at_offset(junction.length)
        nla = vec_to_radians(nlv[:2])

        mli = min(approach_index, len(junction.incoming_lanes) - 1)
        if mli >= 0:
            prev_lane = junction.incoming_lanes[mli]
            plv = prev_lane.vector_at_offset(prev_lane.length - 1)
        else:
            plv = junction.vector_at_offset(0)
        pla = vec_to_radians(plv[:2])

        return min_angles_difference_signed(nla, pla)

    def _higher_priority(
        self,
        junction: RoadMap.Lane,
        dist_to_junction: float,
        traffic_lane: RoadMap.Lane,
        traffic_veh: VehicleState,
        bearing: float,
        foe: RoadMap.Lane,
    ) -> bool:
        owner = self._owner()
        assert owner
        # take into account TLS (don't yield to TL-stopped vehicles)
        # XXX: we currently only determine this for actors we're controlling
        owner = self._owner()
        assert owner
        if traffic_veh.source == owner.source_str and owner._stopped_at_features(
            traffic_veh.actor_id
        ):
            return True

        # Smith vs. Neo
        if traffic_veh.role in (ActorRole.EgoAgent, ActorRole.SocialAgent):
            if self._yield_to_agents == "never":
                return True
            elif self._yield_to_agents == "always":
                return False
            assert (
                self._yield_to_agents == "normal"
            ), f"unknown yield_to_agents value: {self._yield_to_agents}"

        # if already blocking the foes's path then don't yield
        if dist_to_junction <= 0 and self._lane == junction:

            def _in_lane(lane):
                for _, vs, _ in owner._lane_bumpers_cache.get(lane, []):
                    if vs.actor_id == self.actor_id:
                        return True
                return False

            if _in_lane(traffic_lane):
                return True
            if traffic_lane != junction:
                # XXX: might need to go deeper (but don't know their route)...
                for ogtl in traffic_lane.outgoing_lanes:
                    if _in_lane(ogtl):
                        return True

        # Straight > Right > Left priority
        turn_thresh = 0.166 * math.pi
        my_ta = self._turn_angle(junction, self._lane.index)
        their_ta = self._turn_angle(foe, traffic_lane.index)
        if my_ta >= turn_thresh and their_ta < turn_thresh:
            # me left, them not left
            return False
        if abs(my_ta) < turn_thresh and abs(their_ta) >= turn_thresh:
            # me straight, them turning
            return True
        if my_ta <= -turn_thresh:
            # me right
            if their_ta >= turn_thresh:
                # them left
                return True
            if abs(their_ta) < turn_thresh:
                # them straight
                return False

        # Major over minor roads
        my_lanes = len(self._lane.road.lanes)
        their_lanes = len(traffic_lane.road.lanes)
        if my_lanes > their_lanes:
            return True
        # Vehicle to the right (couter clockwise)
        elif my_lanes == their_lanes and bearing > 0:
            return True
        return False

    def _backtrack_until_long_enough(
        self,
        result: Set[RoadMap.Lane],
        lane: RoadMap.Lane,
        cur_length: float,
        min_length: float,
    ):
        result.add(lane)
        cur_length += lane.length
        if cur_length < min_length:
            for il in lane.incoming_lanes:
                if not il.is_drivable:
                    continue
                self._backtrack_until_long_enough(result, il, cur_length, min_length)

    def _handle_junctions(self, dt: float, window: int = 5, max_range: float = 100.0):
        assert max_range > 0, "Max range is used as denominator"
        rl = RoadMap.Route.RouteLane(self._target_lane_win.lane, self._route_ind)
        owner = self._owner()
        assert owner
        l_offset = owner._cached_lane_offset(self._state, self._target_lane_win.lane)
        njl, nj_dist = self._route.next_junction(rl, l_offset)
        if not njl or nj_dist > max_range:
            return
        updated = set()
        my_pos = self._state.pose.point.as_np_array[:2]
        my_heading = self._state.pose.heading
        half_len = 0.5 * self._state.dimensions.length
        hv = radians_to_vec(my_heading)
        my_front = my_pos + half_len * hv
        my_back = my_pos - half_len * hv
        min_range = max_range
        for foe in njl.foes:
            check_lanes = set()
            self._backtrack_until_long_enough(check_lanes, foe, 0, max_range)
            assert check_lanes
            for check_lane in check_lanes:
                if check_lane == self._lane:
                    continue
                handled = set()
                lbc = owner._lane_bumpers_cache.get(check_lane, [])
                for offset, fv, bumper in lbc:
                    if fv.actor_id == self.actor_id:
                        continue
                    # vehicles can be in the bumpers_cache more than once
                    # (for their front and back bumpers, and b/c they may straddle lanes).
                    # We need to check both bumpers for collisions here as (especially
                    # for longer vehicles) the answer can come out differently.
                    if bumper == 0:
                        # don't worry about the "fake back bumper"
                        continue
                    fv_pos = check_lane.from_lane_coord(
                        RefLinePoint(offset)
                    ).as_np_array[:2]
                    f_rb, f_rng = self._bumper_wins_front.add_to_win(
                        fv.actor_id, bumper, fv_pos, my_front, my_heading, dt
                    )
                    b_rb, b_rng = self._bumper_wins_back.add_to_win(
                        fv.actor_id, bumper, fv_pos, my_back, my_heading, dt
                    )
                    updated.add(fv.actor_id)
                    # we will only do something if the potential collider is "ahead" of us...
                    if min(abs(f_rb), abs(b_rb)) >= 0.45 * math.pi:
                        continue
                    est_front_crash = (
                        self._bumper_wins_front.predict_crash_in(fv.actor_id, bumper)
                        if f_rng < max_range
                        else math.inf
                    )
                    est_back_crash = (
                        self._bumper_wins_back.predict_crash_in(fv.actor_id, bumper)
                        if b_rng < max_range
                        else math.inf
                    )
                    if est_front_crash <= est_back_crash:
                        est_crash = est_front_crash
                        rel_bearing = f_rb
                        fv_range = f_rng
                    else:
                        est_crash = est_back_crash
                        rel_bearing = b_rb
                        fv_range = b_rng
                    if est_crash > 60.0:
                        # ignore future crash if after arbitrarily-high threshold of 1 min...
                        continue
                    if fv.actor_id not in handled and not self._higher_priority(
                        njl, nj_dist, check_lane, fv, rel_bearing, foe
                    ):
                        self._logger.debug(
                            f"{self.actor_id} may slow down to avoid collision @ {njl.lane_id} with {fv.actor_id} currently in {check_lane.lane_id}"
                        )
                        rng = nj_dist if nj_dist > 0 else fv_range
                        if rng < min_range:
                            min_range = rng
                        handled.add(fv.actor_id)
                    if check_lane == foe:
                        # we've already picked our lane, but we still update our gaps and ttc
                        # because these are also used by the acceleration PID controller.
                        self._target_lane_win.ttc = min(
                            est_crash, self._target_lane_win.ttc
                        )
                        crash_dist = distance_covered(
                            est_crash, self.speed, self.acceleration
                        )
                        self._target_lane_win.gap = min(
                            crash_dist, self._target_lane_win.gap
                        )
        self._bumper_wins_front.purge_unseen(updated)
        self._bumper_wins_back.purge_unseen(updated)
        self._target_speed *= math.pow(min_range / max_range, 0.75)

    def _find_features_ahead(
        self,
        rl: RoadMap.Route.RouteLane,
        lookahead: float,
        upcoming_feats: List[RoadMap.Feature],
    ):
        lane = rl.lane
        lookahead -= lane.length
        if lane != self._lane:
            upcoming_feats += lane.features
        else:
            lookahead += self._offset
            # make sure the feature is not behind me...
            half_len = 0.5 * self._state.dimensions.length
            my_bb = self._offset - half_len
            for feat in lane.features:
                for pt in feat.geometry:
                    if lane.offset_along_lane(pt) >= my_bb:
                        upcoming_feats.append(feat)
                        break
        if lookahead <= 0:
            return
        nri = rl.road_index + 1
        if nri >= len(self._route.roads):
            return
        for ogl in lane.outgoing_lanes:
            if ogl.road == self._route.roads[nri]:
                nrl = RoadMap.Route.RouteLane(ogl, nri)
                self._find_features_ahead(nrl, lookahead, upcoming_feats)

    def _handle_features_and_signals(self, dt: float):
        my_stopping_dist = stopping_distance(self.speed, self._max_decel)
        lookahead = 2 * my_stopping_dist
        upcoming_feats = []
        rl = RoadMap.Route.RouteLane(self._lane, self._route_ind)
        self._find_features_ahead(rl, lookahead, upcoming_feats)
        # TODO:  check non-fixed-location signals here too
        if not upcoming_feats:
            return
        for feat in upcoming_feats:
            if feat.type == RoadMap.FeatureType.SPEED_BUMP:
                self._target_speed *= 0.5
                continue
            if feat.type == RoadMap.FeatureType.STOP_SIGN:

                def dist_to_stop():
                    dist_to_sign = feat.min_dist_from(self._state.pose.point)
                    lw, _ = self._lane.width_at_offset(self._offset)
                    hdist_to_sign = 0.5 * lw + 2
                    if hdist_to_sign >= dist_to_sign:
                        return 0.0
                    dts = math.sqrt(dist_to_sign**2 - hdist_to_sign**2)
                    dts -= 0.5 * self._state.dimensions.length
                    return max(dts, 0.0)

                if feat.feature_id not in self._stopped_at_feat:
                    dts = dist_to_stop()
                    if self.speed < 0.1 and dts <= self._min_space_cush:
                        self._stopped_at_feat[feat.feature_id] = True
                    self._lane_win.gap = min(dts, self._lane_win.gap)
                elif self._stopped_at_feat[feat.feature_id]:
                    self._waiting_at_feat[feat.feature_id] += dt
                    if self._waiting_at_feat[feat.feature_id] > self._wait_to_restart:
                        del self._waiting_at_feat[feat.feature_id]
                        self._stopped_at_feat[feat.feature_id] = False
                    else:
                        self._lane_win.gap = min(dist_to_stop(), self._lane_win.gap)
                continue
            if feat.is_dynamic:
                owner = self._owner()
                assert owner
                should_stop = owner._signal_state_means_stop(self._lane, feat)
                if should_stop and self.speed == 0.0:
                    self._stopped_at_feat[feat.feature_id] = True
                elif not should_stop and self._stopped_at_feat.get(
                    feat.feature_id, False
                ):
                    del self._stopped_at_feat[feat.feature_id]
                    self._waiting_at_feat[feat.feature_id] = 0.0
                if feat.feature_id in self._waiting_at_feat:
                    if self._waiting_at_feat[feat.feature_id] > self._wait_to_restart:
                        del self._waiting_at_feat[feat.feature_id]
                        continue
                    self._waiting_at_feat[feat.feature_id] += dt
                if should_stop or feat.feature_id in self._waiting_at_feat:
                    # TAI: could decline to do this if my_stopping_dist >> dist_to_stop
                    dist_to_stop = feat.min_dist_from(self._state.pose.point)
                    self._lane_win.gap = min(dist_to_stop, self._lane_win.gap)

    def _check_speed(self, dt: float):
        target_lane = self._target_lane_win.lane
        if target_lane.speed_limit is None:
            self._target_speed = self.speed
            # XXX: on a road like this, we can only slow down
            # (due to curves and traffic) but we never speed back up!
            # TAI: setting the target_speed to some fixed default instead?
        else:
            self._target_speed = target_lane.speed_limit
            self._target_speed *= self._speed_factor
        if self._cutting_in:
            self._target_speed *= self._cutin_slow_down
        self._slow_for_curves()
        if self._target_speed > 0:
            self._handle_features_and_signals(dt)
        if self._target_speed > 0:
            self._handle_junctions(dt)
        if self._target_speed > 0:
            max_speed = float(self._vtype.get("maxSpeed", 55.55))
            if self._target_speed >= max_speed:
                self._target_speed = max_speed

    def _angle_to_lane(self, dt: float) -> float:
        my_heading = self._state.pose.heading
        look_ahead = max(dt * self.speed, 2)
        proj_pt = self._state.pose.position[:2] + look_ahead * radians_to_vec(
            my_heading
        )
        proj_pt = Point(*proj_pt)

        target_lane = self._target_lane_win.lane
        lane_coord = target_lane.to_lane_coord(proj_pt)
        lat_err = lane_coord.t

        target_vec = target_lane.vector_at_offset(lane_coord.s)
        target_heading = vec_to_radians(target_vec[:2])
        heading_delta = min_angles_difference_signed(target_heading, my_heading)

        self._wrong_way = abs(heading_delta) > 0.5 * math.pi
        if self._wrong_way:
            self._logger.warning(
                f"{self.actor_id} going the wrong way for {target_lane.lane_id}.  heading_delta={heading_delta}."
            )
            # slow down so it's easier to turn around
            self._target_speed *= 0.67

        # Here we may also want to take into account speed, accel, inertia, etc.
        # and maybe even self._aggressiveness...

        # magic numbers here were just what looked reasonable in limited testing
        angular_velocity = 3.75 * heading_delta - 1.25 * lat_err

        if not self._wrong_way:
            # add some damping...
            # but only if we're not going the wrong way already! (if we are, try to correct quickly.)
            damping = heading_delta * lat_err
            if damping < 0:
                angular_velocity += 2.2 * np.sign(angular_velocity) * damping
            if self._prev_angular_err is not None:
                angular_velocity -= 0.2 * (heading_delta - self._prev_angular_err[0])
                angular_velocity += 0.2 * (lat_err - self._prev_angular_err[1])
            if abs(angular_velocity) > dt * self._max_angular_velocity:
                angular_velocity = (
                    np.sign(angular_velocity) * dt * self._max_angular_velocity
                )
                # also slow down...
                self._target_speed *= 0.8

        angular_velocity += 0.02 * self._imperfection * (random.random() - 0.5)

        self._prev_angular_err = (heading_delta, lat_err)
        return angular_velocity

    def _near_dest(self, within: float = 0) -> bool:
        if self._lane.road != self._dest_lane.road:
            return False
        dest_lane_offset = self._dest_lane.offset_along_lane(self._state.pose.point)
        dist_left = self._dest_offset - dest_lane_offset
        return dist_left <= within

    def _compute_acceleration(self, dt: float) -> float:
        emergency_decl = float(self._vtype.get("emergencyDecel", 4.5))
        assert emergency_decl >= 0.0
        speed_denom = self.speed
        # usually target_lane == lane, in which case the later terms here are redundant.
        # when they are different, we still care about lane because we are exiting it.
        # Rather than recompute the exit time from our current t-coord, we just
        # use twice the current lane time left as a short-cut.
        time_cush = max(
            min(
                self._target_lane_win.ttc,
                _safe_division(self._target_lane_win.gap, speed_denom),
                self._target_lane_win.time_left,
                self._lane_win.ttc,
                _safe_division(self._lane_win.gap, speed_denom),
                2 * self._lane_win.time_left,
            ),
            1e-13,
        )
        min_time_cush = float(self._vtype.get("tau", 1.0))
        if (
            not self._near_dest(min_time_cush * speed_denom)
            and time_cush < min_time_cush
        ):
            if self.speed > 0:
                severity = 4 * _safe_division(
                    (min_time_cush - time_cush), min_time_cush
                )
                return -emergency_decl * np.clip(severity, 0, 1.0)
            return 0

        space_cush = max(min(self._target_lane_win.gap, self._lane_win.gap), 1e-13)
        if space_cush < self._min_space_cush:
            if self.speed > 0:
                severity = 4 * _safe_division(
                    (self._min_space_cush - space_cush), self._min_space_cush
                )
                return -emergency_decl * np.clip(severity, 0, 1.0)
            return 0

        my_speed, my_acc = self._lane_speed[self._target_lane_win.lane.index]

        # magic numbers / weights here were just what looked reasonable in limited testing
        P = 0.0060 * (self._target_speed - my_speed)
        I = -0.0150 / space_cush + -0.0333 / time_cush
        D = -0.0010 * my_acc
        PID = (P + I + D) / dt

        PID -= 0.02 * self._imperfection * random.random()

        PID = np.clip(PID, -1.0, 1.0)

        if PID > 0:
            return PID * self._max_accel

        return PID * self._max_decel

    def compute_next_state(self, dt: float):
        """Pre-computes the next state for this traffic actor."""
        self._compute_lane_speeds()

        self._pick_lane(dt)
        self._check_speed(dt)

        angular_velocity = self._angle_to_lane(dt)
        acceleration = self._compute_acceleration(dt)

        target_heading = self._state.pose.heading + angular_velocity * dt
        target_heading %= 2 * math.pi
        heading_vec = radians_to_vec(target_heading)
        self._next_linear_acceleration = dt * acceleration * heading_vec
        self._next_speed = self._state.speed + acceleration * dt
        if self._next_speed < 0:
            # don't go backwards
            self._next_speed = 0
        dpos = heading_vec * self.speed * dt
        target_pos = self._state.pose.position + np.append(dpos, 0.0)
        self._next_pose = Pose.from_center(target_pos, Heading(target_heading))

    def step(self, dt: float):
        """Updates to the pre-computed next state for this traffic actor."""
        if self._stranded:
            return
        self._teleporting = False
        self._state.pose = self._next_pose
        self._state.speed = self._next_speed
        self._state.linear_acceleration = self._next_linear_acceleration
        self._state.updated = False
        prev_road_id = self._lane.road.road_id
        self.bbox.cache_clear()

        # if there's more than one lane near us (like in a junction) pick closest one that's in our route
        owner = self._owner()
        assert owner
        nls = owner.road_map.nearest_lanes(
            self._next_pose.point,
            radius=self._state.dimensions.length,
            include_junctions=True,
        )
        if not nls:
            self._logger.warning(
                f"actor {self.actor_id} out-of-lane: {self._next_pose}"
            )
            self._off_route = True
            if self._flow["endless"]:
                self._reroute()
            return
        self._lane = None
        best_ri_delta = None
        next_route_ind = None
        for nl, d in nls:
            try:
                next_route_ind = self._route.road_ids.index(
                    nl.road.road_id, self._route_ind
                )
                self._off_route = False
                ri_delta = next_route_ind - self._route_ind
                if ri_delta <= 1:
                    self._lane = nl
                    best_ri_delta = ri_delta
                    break
                if best_ri_delta is None or ri_delta < best_ri_delta:
                    self._lane = nl
                    best_ri_delta = ri_delta
            except ValueError:
                pass
        if not self._lane:
            self._off_route = True
            self._lane = nls[0][0]
            self._logger.info(
                f"actor {self.actor_id} is off of its route.  now in lane {self._lane.lane_id}."
            )

        road_id = self._lane.road.road_id
        if road_id != prev_road_id:
            self._route_ind += 1 if best_ri_delta is None else best_ri_delta

        self._offset = self._lane.offset_along_lane(self._next_pose.point)
        if self._near_dest():
            if self._lane == self._dest_lane:
                if self._flow["endless"]:
                    self._reroute()
                else:
                    self._done_with_route = True
            else:
                self._off_route = True
                self._logger.info(
                    f"actor {self.actor_id} is beyond end of route in wrong lane ({self._lane.lane_id} instead of {self._dest_lane.lane_id})."
                )

    def _reroute(self):
        if not self._off_route and self._route.road_ids[0] in {
            oid.road_id for oid in self._lane.road.outgoing_roads
        }:
            self._route_ind = -1
            self._logger.debug(
                f"{self.actor_id} will loop around to beginning of its route"
            )
            return
        self._logger.info(f"{self.actor_id} teleporting back to beginning of its route")
        self._teleporting = True
        self._lane, self._offset = self._resolve_flow_pos(
            self._flow, "depart", self._state.dimensions
        )
        position = self._lane.from_lane_coord(RefLinePoint(s=self._offset))
        heading = vec_to_radians(self._lane.vector_at_offset(self._offset)[:2])
        self._state.pose = Pose.from_center(position, Heading(heading))
        start_speed = self._resolve_flow_speed(self._flow)
        if start_speed > 0:
            self._state.speed = start_speed
        self._state.linear_acceleration = np.array((0.0, 0.0, 0.0))
        self._state.updated = False
        self._route_ind = 0
        owner = self._owner()
        assert owner
        if not owner._check_actor_bbox(self):
            self._logger.info(
                f"{self.actor_id} could not teleport back to beginning of its route b/c it was already occupied; just removing it."
            )
            self._done_with_route = True
        elif self._off_route:
            self._off_route = False
