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
import xml.etree.ElementTree as XET
from bisect import bisect_left, bisect_right, insort
from collections import deque
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from cached_property import cached_property
from shapely.affinity import rotate as shapely_rotate
from shapely.geometry import Polygon, box as shapely_box

from .actor_role import ActorRole
from .controllers import ActionSpaceType
from .coordinates import Dimensions, Heading, Point, Pose, RefLinePoint
from .provider import ProviderRecoveryFlags, ProviderState
from .road_map import RoadMap, Waypoint
from .scenario import Scenario
from .traffic_provider import TrafficProvider
from .utils.kinematics import time_to_cover
from .utils.math import (
    min_angles_difference_signed,
    radians_to_vec,
    vec_to_radians,
)
from .vehicle import VEHICLE_CONFIGS, VehicleState


class LocalTrafficProvider(TrafficProvider):
    """A LocalTrafficProvider simulates multiple traffic actors on a generic RoadMap."""

    def __init__(self):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._scenario = None
        self.road_map: RoadMap = None
        self._flows: Dict[str, Dict[str, Any]] = dict()
        self._my_actors: Dict[str, _TrafficActor] = dict()
        self._other_vehicles: Dict[
            str, Tuple[VehicleState, Optional[Sequence[str]]]
        ] = dict()
        self._reserved_areas: Dict[str, Polygon] = dict()
        self._route_lane_lengths: Dict[int, Dict[Tuple[str, int], float]] = dict()
        self._actors_created: int = 0
        self._lane_bumpers_cache: Dict[
            RoadMap.Lane, List[Tuple[float, VehicleState]]
        ] = dict()
        self._offsets_cache: Dict[str, Dict[str, float]] = dict()

    @property
    def action_spaces(self) -> Set[ActionSpaceType]:
        return set()

    def manages_vehicle(self, vehicle_id: str) -> bool:
        return vehicle_id in self._my_actors

    def _cache_route_lengths(self, route: Sequence[str]) -> int:
        # returns a route_key for finding the route in the cache.
        # route_keys shouldn't be exposed/used outside of this file.
        route_key = hash(tuple(route))
        if route_key in self._route_lane_lengths:
            return route_key
        # TAI: could pre-cache curvatures here too (like waypoints) ?
        self._route_lane_lengths[route_key] = dict()

        def _backprop_length(bplane: RoadMap.Lane, length: float, rind: int):
            assert rind >= 0
            rind -= 1
            for il in bplane.incoming_lanes:
                il = il.composite_lane
                ill = self._route_lane_lengths[route_key].get((il.lane_id, rind))
                if ill is not None:
                    self._route_lane_lengths[route_key][(il.lane_id, rind)] = (
                        ill + length
                    )
                    _backprop_length(il, length, rind)

        road = None
        for r_ind, road_id in enumerate(route):
            road = self.road_map.road_by_id(road_id)
            assert road, f"route road '{road_id}' not found in road map"
            for lane in road.lanes:
                lane = lane.composite_lane
                assert (lane.lane_id, r_ind) not in self._route_lane_lengths[route_key]
                _backprop_length(lane, lane.length, r_ind)
                self._route_lane_lengths[route_key][(lane.lane_id, r_ind)] = lane.length
        if not road:
            return route_key
        # give lanes that would form a loop an advantage...
        for lane in road.lanes:
            lane = lane.composite_lane
            for og in lane.outgoing_lanes:
                if (
                    og.road.road_id == route[0]
                    or og.road.composite_road.road_id == route[0]
                ):
                    self._route_lane_lengths[route_key][(lane.lane_id, r_ind)] += 1
        return route_key

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
                route = route.split()
                flow["route"] = route
                flow["begin"] = float(flow["begin"])
                flow["end"] = float(flow["end"])
                if "vehsPerHour" in flow:
                    freq = float(flow["vehsPerHour"])
                    assert freq > 0.0
                    flow["emit_period"] = 3600.0 / freq
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
                flow["route_key"] = self._cache_route_lengths(route)

    def _check_actor_bbox(self, actor: "_TrafficActor") -> bool:
        actor_bbox = actor.bbox(True)
        for reserved_area in self._reserved_areas.values():
            if reserved_area.intersects(actor_bbox):
                return False
        for my_actor in self._my_actors.values():
            if actor != my_actor and my_actor.bbox().intersects(actor_bbox):
                return False
        for other, _ in self._other_vehicles.values():
            if other.bbox.intersects(actor_bbox):
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
        return [other for other, _ in self._other_vehicles.values()]

    @property
    def _all_states(self) -> List[VehicleState]:
        return self._my_actor_states + self._other_vehicle_states

    @property
    def _provider_state(self) -> ProviderState:
        return ProviderState(vehicles=self._my_actor_states)

    def setup(self, scenario: Scenario) -> ProviderState:
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
                insort(lbc, (back_offset, ovs))
            if front_lane:
                front_offset = front_lane.offset_along_lane(front)
                lbc = self._lane_bumpers_cache.setdefault(front_lane, [])
                insort(lbc, (front_offset, ovs))
            if front_lane and back_lane != front_lane:
                # it's changing lanes, don't misjudge the target lane...
                fake_back_offset = front_lane.offset_along_lane(back)
                insort(self._lane_bumpers_cache[front_lane], (fake_back_offset, ovs))

    def _cached_lane_offset(self, vs: VehicleState, lane: RoadMap.Lane):
        lane_offsets = self._offsets_cache.setdefault(vs.vehicle_id, dict())
        return lane_offsets.setdefault(
            lane.lane_id, lane.offset_along_lane(vs.pose.point)
        )

    def step(self, actions, dt: float, elapsed_sim_time: float) -> ProviderState:
        self._add_actors_for_time(elapsed_sim_time, dt)
        for other, _ in self._other_vehicles.values():
            if other.vehicle_id in self._reserved_areas:
                del self._reserved_areas[other.vehicle_id]

        # precompute nearest lanes and offsets for all vehicles and cache
        # (this prevents having to do it O(ovs^2) times)
        self._create_actor_caches()

        # Do state update in two passes so that we don't use next states in the
        # computations for actors encountered later in the iterator.
        for actor in self._my_actors.values():
            actor.compute_next_state(dt)

        dones = []
        remap_ids: Dict[str, str] = dict()
        for actor_id, actor in self._my_actors.items():
            actor.step(dt)
            if actor.finished_route or actor.off_route:
                dones.append(actor.actor_id)
            elif actor.teleporting:
                # pybullet doesn't like it when a vehicle jumps from one side of the map to another,
                # so we need to give teleporting vehicles a new id and thus a new chassis.
                actor.bump_id()
                remap_ids[actor_id] = actor.actor_id
        for actor_id in dones:
            del self._my_actors[actor_id]
        for orig_id, new_id in remap_ids.items():
            self._my_actors[new_id] = self._my_actors[orig_id]
            del self._my_actors[orig_id]

        return self._provider_state

    def sync(self, provider_state: ProviderState):
        missing = self._my_actors.keys() - {
            psv.vehicle_id for psv in provider_state.vehicles
        }
        for left in missing:
            self._logger.warning(
                f"locally provided actor '{left}' disappeared from simulation"
            )
            del self._my_actors[left]
        hijacked = self._my_actors.keys() & {
            psv.vehicle_id
            for psv in provider_state.vehicles
            if psv.source != self.source_str
        }
        for jack in hijacked:
            self.stop_managing(jack)
        self._other_vehicles = dict()
        for vs in provider_state.vehicles:
            my_actor = self._my_actors.get(vs.vehicle_id)
            if my_actor:
                assert vs.source == self.source_str
                # here we override our state with the "consensus" state...
                # (Note: this is different from what Sumo does;
                # we're allowing for true "harmonization" if necessary.
                # "You may say that I'm a dreamer, but I'm not the only one,
                # I hope that one day you'll join us, and the world will
                # be as one." ;)
                my_actor.state = vs
            else:
                assert vs.source != self.source_str
                self._other_vehicles[vs.vehicle_id] = (vs, None)

    def reset(self):
        # Unify interfaces with other providers
        pass

    def teardown(self):
        self._my_actors = dict()
        self._other_vehicles = dict()
        self._reserved_areas = dict()

    def destroy(self):
        pass

    def stop_managing(self, vehicle_id: str):
        # called when agent hijacks this vehicle
        assert (
            vehicle_id in self._my_actors
        ), f"stop_managing() called for non-tracked vehicle id '{vehicle_id}'"
        del self._my_actors[vehicle_id]

    def reserve_traffic_location_for_vehicle(
        self,
        vehicle_id: str,
        reserved_location: Polygon,
    ):
        self._reserved_areas[vehicle_id] = reserved_location

    def vehicle_collided(self, vehicle_id: str):
        # TAI:  consider removing the vehicle?
        # If collidee(s) include(s) an EgoAgent, it will likely be marked "done" and things will end.
        # (But this is not guaranteed depending on the criteria that were set.)
        # Probably the most realistic thing we can do is leave the vehicle sitting in the road, blocking traffic!
        # (... and then add a "rubber-neck mode" for all nearby vehicles?! ;)
        # Let's do that for now, but we should also consider just removing the vehicle.
        traffic_actor = self._my_actors.get(vehicle_id)
        if not traffic_actor:
            # guess we already removed it for some other reason (off route?)
            return
        traffic_actor.stay_put()

    def update_route_for_vehicle(self, vehicle_id: str, new_route_roads: Sequence[str]):
        traffic_actor = self._my_actors.get(vehicle_id)
        if traffic_actor:
            route_key = self._cache_route_lengths(new_route_roads)
            traffic_actor.update_route(route_key, new_route_roads)
            return
        other = self._other_vehicles.get(vehicle_id)
        if other:
            self._other_vehicles[vehicle_id] = (other[0], new_route_roads)
            return
        assert False, f"unknown vehicle_id: {vehicle_id}"

    def vehicle_dest_road(self, vehicle_id: str) -> Optional[str]:
        traffic_actor = self._my_actors.get(vehicle_id)
        if traffic_actor:
            return traffic_actor.route[-1]
        other = self._other_vehicles.get(vehicle_id)
        if other:
            return other[1][-1] if other[1] else None
        assert False, f"unknown vehicle_id: {vehicle_id}"

    def can_accept_vehicle(self, state: VehicleState) -> bool:
        return state.role == ActorRole.Social or state.role == ActorRole.Unknown

    def add_vehicle(
        self,
        provider_vehicle: VehicleState,
        route: Optional[Sequence[RoadMap.Route]] = None,
    ):
        provider_vehicle.source = self.source_str
        provider_vehicle.role = ActorRole.Social
        xfrd_actor = _TrafficActor.from_state(provider_vehicle, self, route)
        self._my_actors[xfrd_actor.actor_id] = xfrd_actor
        if xfrd_actor.actor_id in self._other_vehicles:
            del self._other_vehicles[xfrd_actor.actor_id]
        self._logger.info(
            f"traffic actor {xfrd_actor.actor_id} transferred to {self.source_str}."
        )


class _TrafficActor:
    """Simulates a vehicle managed by the LocalTrafficProvider."""

    def __init__(self, flow: Dict[str, Any], owner: LocalTrafficProvider):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._logger.setLevel(logging.INFO)

        self._owner = owner
        self._state = None
        self._flow: Dict[str, Any] = flow
        self._vtype: Dict[str, Any] = flow["vtype"]
        self._route_ind: int = 0
        self._done_with_route: bool = False
        self._off_route: bool = False
        self._route: List[str] = flow["route"]
        self._route_key: int = flow["route_key"]
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

        self._cutting_into = None
        self._in_front_after_cutin_secs = 0
        self._cutin_hold_secs = 3
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
        self._dogmatic = bool(self._vtype.get("lcDogmatic", False))

        self._max_angular_velocity = 26  # arbitrary, based on limited testing
        self._prev_angular_err = None

        self._owner._actors_created += 1

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
        new_actor._state = VehicleState(
            vehicle_id=f"{new_actor._vtype['id']}-{new_actor._owner._actors_created}",
            pose=Pose.from_center(position, Heading(heading)),
            dimensions=dimensions,
            vehicle_type=vehicle_type,
            vehicle_config_type=vclass,
            speed=init_speed,
            linear_acceleration=np.array((0.0, 0.0, 0.0)),
            source=new_actor._owner.source_str,
            role=ActorRole.Social,
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
        route: Optional[RoadMap.Route],
    ):
        """Factory to construct a _TrafficActor object from an existing VehiclState object."""
        cur_lane = owner.road_map.nearest_lane(state.pose.point)
        if not route or not route.roads:
            route = owner.road_map.random_route(starting_road=cur_lane.road)
        route_roads = [road.road_id for road in route.roads]
        route_key = owner._cache_route_lengths(route_roads)
        flow = dict()
        flow["vtype"] = dict()
        flow["route"] = route_roads
        flow["route_key"] = route_key
        flow["arrivalLane"] = f"{cur_lane.index}"  # XXX: assumption!
        flow["arrivalPos"] = "max"
        flow["departLane"] = f"{cur_lane.index}"  # XXX: assumption!
        flow["departPos"] = "0"
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
        road_id = self._route[0] if depart_arrival == "depart" else self._route[-1]
        road = self._owner.road_map.road_by_id(road_id)
        if not road:
            raise Exception(f"{base_err}:  road_id '{road_id}' not in map.")
        lane_ind = int(flow[f"{depart_arrival}Lane"])
        if not 0 <= lane_ind < len(road.lanes):
            raise Exception(
                f"{base_err}:  lane index {lane_ind} invalid for road_id '{road_id}'."
            )
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
                f"{base_err}:  starting offset {offset} invalid for road_id '{road_id}'."
            )
        # convert to composite system...
        target_pt = lane.from_lane_coord(RefLinePoint(s=offset))
        lane = lane.composite_lane
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
        return self._state.vehicle_id

    def bump_id(self):
        """Changes the id of a teleporting vehicle."""
        mm = re.match(r"([^_]+)(_(\d+))?$", self._state.vehicle_id)
        assert mm
        ver = int(mm.group(3)) if mm.group(3) else 0
        self._state.vehicle_id = f"{mm.group(1)}_{ver + 1}"

    @property
    def route(self) -> List[str]:
        """The route (sequence of road_ids) this actor will attempt to take."""
        return self._route

    def update_route(self, route_key: int, route: List[str]):
        """Update the route (sequence of road_ids) this actor will attempt to take.
        A unique route_key is provided for referencing the route cache in he owner provider."""
        self._route = route
        self._route_key = route_key
        self._dest_lane, self._dest_offset = self._resolve_flow_pos(
            self._flow, "arrival", self._state.dimensions.length
        )
        self._route_ind = 0

    def stay_put(self):
        """Tells this actor to stop acting and remain where it is indefinitely."""
        if not self._stranded:
            self._logger.info(f"{self.actor_id} stranded")
            self._stranded = True

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

    class _LaneWindow:
        def __init__(
            self,
            lane: RoadMap.Lane,
            time_left: float,
            ttre: float,
            gap: float,
            lane_coord: RefLinePoint,
            agent_gap: Optional[float],
        ):
            self.lane = lane
            self.time_left = time_left
            self.adj_time_left = time_left  # could eventually be negative
            self.ttre = ttre  # time until we'd get rear-ended
            self.gap = gap  # just the gap ahead (in meters)
            self.lane_coord = lane_coord
            self.agent_gap = agent_gap

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
                return 1.0 / math.sin(theta)
            # here we correct for the local road curvature (which affects how far we must travel)...
            T = self.radius / self.width
            assert (
                abs(T) > 1.5
            ), f"abnormally high curvature?  radius={self.radius}, width={self.width} at offset {self.lane_coord.s} of lane {self.lane.lane_id}"
            if to_index > self.lane.index:
                se = T * (T - 1)
                return math.sqrt(
                    2 * (se + 0.5 - se * math.cos(1 / (math.tan(theta) * (T - 1))))
                )
            se = T * (T + 1)
            return math.sqrt(
                2 * (se + 0.5 - se * math.cos(1 / (math.tan(theta) * (T + 1))))
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
        self, lane: RoadMap.Lane, dte: float, route_lens, rind: int
    ) -> Tuple[float, Optional[VehicleState]]:
        nv_ahead_dist = math.inf
        nv_ahead_vs = None
        rind += 1
        for ogl in lane.outgoing_lanes:
            ogl = ogl.composite_lane
            len_to_end = route_lens.get((ogl.lane_id, rind))
            if len_to_end is None:
                continue
            lbc = self._owner._lane_bumpers_cache.get(ogl)
            if lbc:
                fi = 0
                while fi < len(lbc):
                    ogl_offset, ogl_vs = lbc[fi]
                    if ogl_vs.vehicle_id != self.actor_id:
                        break
                    fi += 1
                if fi == len(lbc):
                    continue
                ogl_dist = dte - (len_to_end - ogl_offset)
                if ogl_dist < nv_ahead_dist:
                    nv_ahead_dist = ogl_dist
                    nv_ahead_vs = ogl_vs
                continue
            ogl_dist, ogl_vs = self._find_vehicle_ahead_on_route(
                ogl, dte, route_lens, rind
            )
            if ogl_dist < nv_ahead_dist:
                nv_ahead_dist = ogl_dist
                nv_ahead_vs = ogl_vs
        return nv_ahead_dist, nv_ahead_vs

    def _find_vehicle_ahead(
        self, lane: RoadMap.Lane, my_offset: float, search_start: float
    ) -> Tuple[float, Optional[VehicleState]]:
        lbc = self._owner._lane_bumpers_cache.get(lane)
        if lbc:
            lane_spot = bisect_right(lbc, (search_start, self._state))
            # if we're at an angle to the lane, it's possible for the
            # first thing we hit to be our own entries in the bumpers cache,
            # which we need to skip.
            while (
                lane_spot < len(lbc) and self.actor_id == lbc[lane_spot][1].vehicle_id
            ):
                lane_spot += 1
            if lane_spot < len(lbc):
                lane_offset, nvs = lbc[lane_spot]
                assert lane_offset >= search_start
                if lane_offset > my_offset:
                    return lane_offset - my_offset, nvs
                return 0, nvs
        route_lens = self._owner._route_lane_lengths[self._route_key]
        route_len = route_lens.get((lane.lane_id, self._route_ind), lane.length)
        my_dist_to_end = route_len - my_offset
        return self._find_vehicle_ahead_on_route(
            lane, my_dist_to_end, route_lens, self._route_ind
        )

    def _find_vehicle_behind(
        self, lane: RoadMap.Lane, my_offset: float, search_start: float
    ) -> Tuple[float, Optional[VehicleState]]:
        lbc = self._owner._lane_bumpers_cache.get(lane)
        if lbc:
            lane_spot = bisect_left(lbc, (search_start, self._state))
            # if we're at an angle to the lane, it's possible for the
            # first thing we hit to be our own entries in the bumpers cache,
            # which we need to skip.
            while lane_spot > 0 and self.actor_id == lbc[lane_spot - 1][1].vehicle_id:
                lane_spot -= 1
            if lane_spot > 0:
                lane_offset, bv_vs = lbc[lane_spot - 1]
                assert lane_offset <= search_start
                if lane_offset < my_offset:
                    return my_offset - lane_offset, bv_vs
                return 0, bv_vs
        # only look back one lane...
        bv_behind_dist = math.inf
        bv_behind_vs = None
        for inl in lane.incoming_lanes:
            inl = inl.composite_lane
            plbc = self._owner._lane_bumpers_cache.get(inl)
            if not plbc:
                continue
            bi = -1
            bv_offset, bv_vs = plbc[bi]
            while bi > -len(plbc) and bv_vs.vehicle_id == self.actor_id:
                bi -= 1
                bv_offset, bv_vs = plbc[bi]
            bv_dist = inl.length - bv_offset
            if bv_dist < bv_behind_dist:
                bv_behind_dist = bv_dist
                bv_behind_vs = bv_vs
        return my_offset + bv_behind_dist, bv_behind_vs

    def _compute_lane_window(self, lane: RoadMap.Lane):
        lane = lane.composite_lane
        lane_coord = lane.to_lane_coord(self._state.pose.point)
        my_offset = lane_coord.s
        my_speed, my_acc = self._lane_speed[lane.index]
        my_route_lens = self._owner._route_lane_lengths[self._route_key]
        path_len = my_route_lens.get((lane.lane_id, self._route_ind), lane.length)
        path_len -= my_offset
        lane_time_left = path_len / self.speed if self.speed else math.inf

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
            min(lane_time_left, lane_ttc),
            lane_ttre,
            ahead_dist,
            lane_coord,
            behind_dist if bv_vs and bv_vs.role == ActorRole.EgoAgent else None,
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
        min_idx = min(target_idx, my_idx + 1)
        max_idx = max(target_idx + 1, my_idx)
        cross_time = self._lane_windows[my_idx].exit_time(
            self.speed, target_idx, self.acceleration
        )
        for i in range(min_idx, max_idx):
            lw = self._lane_windows[i]
            lct = lw.crossing_time_at_speed(target_idx, self.speed, self.acceleration)
            if i == target_idx:
                lct *= 0.75
            cross_time += lct
        # note: we *could* be more clever and use cross_time for each lane separately
        # to try to thread our way through the gaps independently... nah.
        for i in range(min_idx, max_idx):
            lw = self._lane_windows[i]
            if min(lw.time_left, lw.ttre) <= cross_time:
                return cross_time, False
        return cross_time, True

    def _should_cutin(self, lw: _LaneWindow) -> bool:
        if lw.lane.index == self._lane.index:
            return False
        min_gap = self._target_cutin_gap / self._aggressiveness
        max_gap = self._target_cutin_gap + 2
        if (
            min_gap < lw.agent_gap < max_gap
            and self._crossing_time_into(lw.lane.index)[1]
        ):
            return random.random() < self._cutin_prob
        return False

    def _pick_lane(self, dt: float):
        self._compute_lane_windows()
        my_idx = self._lane.index
        self._lane_win = my_lw = self._lane_windows[my_idx]
        best_lw = self._lane_windows[my_idx]
        for l in range(len(self._lane_windows)):
            idx = (my_idx + l) % len(self._lane_windows)
            if l > 0 and not self._crossing_time_into(idx)[1]:
                continue
            lw = self._lane_windows[idx]
            if (
                lw.lane == self._dest_lane
                and lw.lane_coord.s + lw.gap >= self._dest_offset
            ):
                # TAI: speed up or slow down as appropriate if _crossing_time_into() was False
                best_lw = lw
                if not self._dogmatic:
                    break
            if (
                self._cutting_into
                and self._cutting_into.lane.index < len(self._lane_windows)
                and self._crossing_time_into(self._cutting_into.lane.index)[1]
            ):
                best_lw = self._cutting_into
                if self._cutting_into.lane != self._lane:
                    break
                self._in_front_after_cutin_secs += dt
                if self._in_front_after_cutin_secs < self._cutin_hold_secs:
                    break
            self._cutting_into = None
            self._in_front_secs = 0
            if lw.agent_gap and self._should_cutin(lw):
                best_lw = lw
                self._cutting_into = lw
            elif lw.adj_time_left > best_lw.adj_time_left or (
                lw.adj_time_left == best_lw.adj_time_left
                and (
                    (lw.lane == self._dest_lane and self._offset < self._dest_offset)
                    or (lw.ttre > best_lw.ttre and idx < best_lw.lane.index)
                )
            ):
                best_lw = lw
        if best_lw.lane != self._lane and not self._cutting_into:
            self._cutting_into = best_lw
        self._target_lane_win = best_lw

    def _compute_lane_speeds(self):
        def _get_radius(lane: RoadMap.Lane) -> float:
            l_offset = self._owner._cached_lane_offset(self._state, lane)
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
                    ratio = l_radius / my_radius
                    if ratio < 0:
                        ratio = 1.0
            self._lane_speed[l.index] = (ratio * self.speed, ratio * self.acceleration)

    def _slow_for_curves(self):
        # XXX:  this may be too expensive.  if so, we'll need to precompute curvy spots for routes
        lookahead = math.ceil(1 + math.log(self._target_speed))
        # we round the offset in an attempt to reduce the unique hits on the LRU caches...
        rounded_offset = round(self._offset)
        radius = self._lane.curvature_radius_at_offset(rounded_offset, lookahead)
        # pi/2 radian right turn == 6 m/s with radius 10.5 m
        # TODO:  also depends on vehicle type (traction, length, etc.)
        self._target_speed = min(abs(radius) * 0.5714, self._target_speed)

    def _check_speed(self):
        target_lane = self._target_lane_win.lane
        if target_lane.speed_limit is None:
            self._target_speed = self.speed
            self._slow_for_curves()
            return
        self._target_speed = target_lane.speed_limit
        self._target_speed *= self._speed_factor
        # TAI: consider going faster the further left the target lane
        # ... or let scenario creator manage this via flows?
        # self._target_speed *= 1.0 + .05 * self._target_lane_win.lane.index
        self._slow_for_curves()
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
        speed_denom = self.speed if self.speed else 0.1
        time_cush = max(
            min(
                self._target_lane_win.time_left,
                self._target_lane_win.gap / speed_denom,
                self._lane_win.time_left,
                self._lane_win.gap / speed_denom,
            ),
            0,
        )
        min_time_cush = float(self._vtype.get("tau", 1.0))
        if (
            not self._near_dest(min_time_cush * speed_denom)
            and time_cush < min_time_cush
        ):
            if self.speed > 0:
                severity = 4 * (min_time_cush - time_cush) / min_time_cush
                return -emergency_decl * np.clip(severity, 0, 1.0)
            return 0

        space_cush = max(min(self._target_lane_win.gap, self._lane_win.gap), 0)
        if space_cush < self._min_space_cush:
            if self.speed > 0:
                severity = (
                    4 * (self._min_space_cush - space_cush) / self._min_space_cush
                )
                return -emergency_decl * np.clip(severity, 0, 1.0)
            return 0

        my_speed, my_acc = self._lane_speed[self._target_lane_win.lane.index]

        # magic numbers / weights here were just what looked reasonable in limited testing
        P = 0.0060 * (self._target_speed - my_speed)
        I = -0.0150 / space_cush
        D = -0.0010 * my_acc
        PID = (P + I + D) / dt

        PID -= 0.02 * self._imperfection * random.random()

        PID = np.clip(PID, -1.0, 1.0)

        if PID > 0:
            max_accel = float(self._vtype.get("accel", 2.6))
            assert max_accel >= 0.0
            return PID * max_accel

        max_decel = float(self._vtype.get("decel", 4.5))
        assert max_decel >= 0.0
        return PID * max_decel

    def compute_next_state(self, dt: float):
        """Pre-computes the next state for this traffic actor."""
        self._compute_lane_speeds()

        self._pick_lane(dt)
        self._check_speed()

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
        nls = self._owner.road_map.nearest_lanes(
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
                next_route_ind = self._route.index(nl.road.road_id, self._route_ind)
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
        self._lane = self._lane.composite_lane

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
        if not self._off_route and self._route[0] in {
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
        if not self._owner._check_actor_bbox(self):
            self._logger.warning(
                f"{self.actor_id} could not teleport back to beginning of its route b/c it was already occupied; just removing it."
            )
            self._done_with_route = True
        elif self._off_route:
            self._off_route = False
