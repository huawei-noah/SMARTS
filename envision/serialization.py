# MIT License
# 
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
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

# TODO: Output the following:
# _x = float
# _y = float
# _z = float
# _pos=Tuple[_x,_y,_z]
# _heading = float
# _id = int
# _pose = Tuple[_x, _y, _z, _heading]
# from gym.spaces import Space
# _Layout=Space
# _Waypoint=[
#     *_pose,
#     _id, # lane_id
#     float, # lane_width_at_waypoint
#     float, # speed_limit
#     int, # lane_index
# ]
# _Traffic = [
#     int, # actor_id # mapped to id in scenario level state
#     int, # lane_id # mapped to name in scenario level state
#     *_pose, # [x,y,z,heading]
#     float, #speed
#     Optional[Sequence[_Waypoint]], # waypoint_paths
#     Optional[Sequence[_pos]], # point cloud positions [x,y,x2,y2,...,xn,yn]
#     Optional[Sequence[_pos]], # mission route geometry positions [x,y,x2,y2,...,xn,yn]
#     int, #actor_type # Send once
#     int, #vehicle_type # Send once
#     # bitfield, events
#     # remove name
# ]
# _Geometry = Tuple[Sequence[Tuple[_x, _y]], _pose]
# _State = [
#     int, # frame time
#     str, # scenario id
#     Optional[str], # scenario name # send once
#     *Sequence[Tuple[_id, Union[_Geometry, _pose]]], # should be sent updates only Optional[id,geometry] optional[pos,heading]
#     *Sequence[_Traffic],
# ]
# _Payload = [
#     [
#         *_State,
#         Dict[_id, str], # added vehicle ids
#         Sequence[_id], # removed vehicle ids
#         Dict[Tuple[_id, str]], # added agent ids
#         Sequence[_id], # removed agent ids
#         Dict[Tuple[_id, str]], # added lane ids
#         Sequence[_id, str], # removed lane ids
#     ],
#     Optional[_Layout]
# ]

from dataclasses import dataclass
from enum import IntEnum, unique
from types import TracebackType
from typing import (
    Any,
    Callable,
    ContextManager,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Type,
)

from envision.types import State, TrafficActorState, TrafficActorType, VehicleType
from smarts.core.events import Events


@unique
class Context(IntEnum):
    NONE = 0  # implemented
    """No special operation. Value will be sent as is."""
    REDUCE = 1
    """Convert the value to an integer and centralize the value into a list."""
    DELTA = 2
    """Send value only if it has changed."""
    FLATTEN = 4  # implemented
    """Convert value from list or dataclass to higher hierachy."""
    OPTIONAL = 8
    """This is sent only sometimes."""
    ONCE = 16
    """This will be sent only the first time it occurs."""
    LOCAL_DELTA = 32
    """This value will only be sent."""
    DELTA_ALTERNATE = 64
    """Similar to DELTA, this value will be sent as the base value the first time seen then as an alternate."""


_serialization_map: Dict[Type, Callable[[Any, "EnvisionDataFormatter"], None]] = {}

_primitives = {int, float, str, VehicleType, TrafficActorType}


@dataclass
class ReductionContext:
    current_count: int = 0
    items = {}
    removed = []

    def resolve_value(self, v):
        cc = self.current_count
        reduce = self.items.setdefault(v, cc)
        if self.current_count == reduce:
            self.current_count += 1
        return reduce


class EnvisionDataFormatter:
    def __init__(
        self,
        id,
        context: "EnvisionDataFormatter" = None,
        serializer: Callable[[list], Any] = lambda d: d,
    ):
        # self.seen_objects = context.seen_objects if context else set()
        self.layer_id: Any = id
        self.parent_context: EnvisionDataFormatter = context
        self._children: Set[EnvisionDataFormatter] = set()
        self._data: List[Any] = []
        self._reduction_context = ReductionContext()
        self._serializer = serializer

        if self.parent_context:
            self.parent_context._add_child(self)

    def _add_child(self, other):
        self._children.add(other)

    def add_any(self, value: Any):
        if type(value) in _primitives:
            self.add_primitive(value)
        else:
            self.add(value, None)

    def add_primitive(self, value: Any):
        # TODO prevent cycles
        # if isinstance(value, (Object)) and value in self.seen_objects:
        #     return

        f = _serialization_map.get(type(value))
        if f:
            f(value, self)
        else:
            self._data.append(value)

    def add(
        self,
        value: Any,
        id_: str,
        op: Context = Context.NONE,
        select: Callable[[Any], Any] = None,
        alternate: Callable[[Any], Any] = None,
    ):
        outval = value
        if op & Context.REDUCE:
            outval = ReductionContext.resolve_value(self._reduction_context, outval)
        if op & Context.FLATTEN:
            if not isinstance(outval, (Sequence)):
                assert False, "Must use flatten with Sequence or dataclass"
            for e in outval:
                self.add_primitive(e)
        else:
            self.add_primitive(outval)

    class DataFormatterLayer(ContextManager):
        def __init__(self, context: "EnvisionDataFormatter") -> None:
            super().__init__()
            self._context = context
            self._upper_layer_data = context._data

        def __enter__(self):
            super().__enter__()
            self._context._data = []
            return self

        def __exit__(
            self,
            __exc_type: Optional[Type[BaseException]],
            __exc_value: Optional[BaseException],
            __traceback: Optional[TracebackType],
        ) -> Optional[bool]:
            d = self._context._data
            self._context._data = self._upper_layer_data
            self._context.add_primitive(d)
            return super().__exit__(__exc_type, __exc_value, __traceback)

    def layer(self):
        return self.DataFormatterLayer(self)

    def resolve(self):
        if self._reduction_context.current_count > 0:
            self._data.append({v: k for k, v in self._reduction_context.items.items()})
            self._data.append(self._reduction_context.removed)
        return self._serializer(self._data)


def _format_traffic_actor(obj, context: EnvisionDataFormatter):
    assert type(obj) is TrafficActorState
    context.add(obj.actor_id, "actor_id", op=Context.REDUCE)
    context.add(obj.lane_id, "lane_id", op=Context.DELTA | Context.REDUCE)
    context.add(obj.position, "position", op=Context.FLATTEN)
    context.add_primitive(obj.heading)
    context.add_primitive(obj.speed)
    context.add(obj.events, "events", op=Context.LOCAL_DELTA)
    context.add(obj.score, "score", op=Context.OPTIONAL)
    context.add(obj.waypoint_paths, "waypoint_paths", op=Context.OPTIONAL)
    context.add(obj.driven_path, "driven_path", op=Context.OPTIONAL)
    context.add(obj.point_cloud, "point_cloud", op=Context.OPTIONAL)
    context.add(
        obj.mission_route_geometry or [], "mission_route_geometry", op=Context.OPTIONAL
    )
    context.add(obj.actor_type, "actor_type", op=Context.ONCE)
    context.add(obj.vehicle_type, "vehicle_type", op=Context.ONCE)


def _format_state(obj: State, context: EnvisionDataFormatter):
    assert type(obj) is State
    context.add(obj.frame_time, "frame_time")
    context.add(obj.scenario_id, "scenario_id")
    context.add(obj.scenario_name, "scenario_name", op=Context.ONCE)
    with context.layer():
        for t in obj.traffic:
            with context.layer():
                context.add(t, "traffic")
    context.add(
        obj.bubbles,
        "bubbles",
        select=lambda bbl: (bbl.geometry, bbl.pose),
        alternate=lambda bbl: bbl.pose,
        op=Context.DELTA_ALTERNATE,
    )  # On delta use alternative
    # context.add(obj.ego_agent_ids, "ego_agent_ids", op=Context.DELTA)


def _format_vehicle_type(obj: VehicleType, context: EnvisionDataFormatter):
    t = type(obj)
    assert t is VehicleType
    mapping = {
        VehicleType.Bus: 0,
        VehicleType.Coach: 1,
        VehicleType.Truck: 2,
        VehicleType.Trailer: 3,
        VehicleType.Car: 4,
    }
    context.add_primitive(mapping[obj])


def _format_traffic_actor_type(obj: TrafficActorType, context: EnvisionDataFormatter):
    t = type(obj)
    assert t is TrafficActorType
    mapping = {
        TrafficActorType.SocialVehicle: 0,
        TrafficActorType.SocialAgent: 1,
        TrafficActorType.Agent: 2,
    }
    context.add_primitive(mapping[obj])


def _format_events(obj: Events, context: EnvisionDataFormatter):
    t = type(obj)
    assert t is Events
    context.add_primitive(tuple(obj))


_serialization_map[TrafficActorState] = _format_traffic_actor
_serialization_map[State] = _format_state
_serialization_map[VehicleType] = _format_vehicle_type
_serialization_map[TrafficActorType] = _format_traffic_actor_type
_serialization_map[Events] = _format_events
