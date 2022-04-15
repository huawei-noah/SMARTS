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

from enum import IntEnum, unique
from types import TracebackType
from typing import (
    Any,
    Callable,
    ContextManager,
    Dict,
    Generator,
    Hashable,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Type,
    Union,
)

from envision.types import State, TrafficActorState, TrafficActorType, VehicleType
from smarts.core.events import Events
from smarts.core.road_map import Waypoint


@unique
class Operation(IntEnum):
    NONE = 0
    """No special operation. Value will be sent as is."""
    REDUCE = 1
    """Convert the value to an integer and centralize the value into a list."""
    DELTA = 2  # TODO implement
    """Send value only if it has changed."""
    FLATTEN = 4
    """Convert value from list or dataclass to higher hierachy."""
    OPTIONAL = 8
    """Sending this value is togglable by option."""
    ONCE = 16  # TODO implement
    """This will be sent only if it was not sent in the previous reduction."""
    DELTA_ALTERNATE = 64  # TODO implement
    """Similar to DELTA, this value will be sent as the base value the first time seen then as an alternate."""


_formatter_map: Dict[Type, Callable[[Any, "EnvisionDataFormatter"], None]] = {}
_sequence_formatter_map: Dict[Type, Callable[[Any, "EnvisionDataFormatter"], None]] = {}

_primitives = {int, float, str, VehicleType, TrafficActorType}


class ReductionContext:
    def __init__(
        self,
        mapping: Optional[Dict[Hashable, int]] = None,
        removed: Optional[List[int]] = None,
    ) -> None:
        self.current_id = 0
        self._mapping = mapping or {}
        self.removed = removed or []

    @property
    def has_values(self):
        return len(self._mapping) > 0

    @staticmethod
    def resolve_mapping(rc: "ReductionContext"):
        return {k: v for _, (k, v) in rc._mapping.items()}

    @staticmethod
    def resolve_value(rc: "ReductionContext", value: Hashable) -> int:
        cc = rc.current_id
        reduce, _ = rc._mapping.setdefault(hash(value), (cc, value))
        if rc.current_id == reduce:
            rc.current_id += 1
        return reduce


class EnvisionDataFormatter:
    def __init__(
        self,
        id,
        formatter: "EnvisionDataFormatter" = None,
        serializer: Callable[[list], Any] = lambda d: d,
    ):
        # self.seen_objects = context.seen_objects if context else set()
        self.id: Any = id
        self.parent_context: EnvisionDataFormatter = formatter
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

        f = _formatter_map.get(type(value))
        if f:
            f(value, self)
        else:
            self._data.append(value)

    def add(
        self,
        value: Any,
        id_: str,
        op: Operation = Operation.NONE,
        select: Callable[[Any], Any] = None,
        alternate: Callable[[Any], Any] = None,
    ):
        outval = value
        f = _sequence_formatter_map.get(type(value))
        if op & Operation.REDUCE:
            outval = ReductionContext.resolve_value(self._reduction_context, outval)
        if op & Operation.FLATTEN:
            if not isinstance(outval, (Sequence)):
                assert False, "Must use flatten with Sequence or dataclass"
            for e in outval:
                self.add_primitive(e)
        elif f:
            f(value, self)
        else:
            self.add_primitive(outval)

    class DataFormatterLayer(ContextManager, Iterator):
        def __init__(
            self,
            context: "EnvisionDataFormatter",
            iterable: Iterable,
            op: Operation,
        ) -> None:
            super().__init__()
            self._context = context
            self._upper_layer_data = context._data
            self._operation = op

            def empty_gen():
                return
                yield

            self._iterable: Generator[Any, None, None] = (
                (v for v in iterable) if iterable else empty_gen()
            )

        def __enter__(self):
            super().__enter__()
            self._context._data = []
            return self._iterable

        def __exit__(
            self,
            __exc_type: Optional[Type[BaseException]],
            __exc_value: Optional[BaseException],
            __traceback: Optional[TracebackType],
        ) -> Optional[bool]:
            d = self._context._data
            self._context._data = self._upper_layer_data
            self._context.add(d, "", op=self._operation)
            return super().__exit__(__exc_type, __exc_value, __traceback)

        def __iter__(self) -> Iterator[Any]:
            super().__iter__()
            self._context._data = []
            return self

        def __next__(self) -> Any:
            try:
                n = next(self._iterable)
                return n
            except StopIteration:
                d = self._context._data
                self._context._data = self._upper_layer_data
                self._context.add_primitive(d)
                raise

    def layer(self, iterable: Iterable = None, op: Operation = Operation.NONE):
        return self.DataFormatterLayer(self, iterable, op)

    def resolve(self):
        if self._reduction_context.has_values:
            self._data.append(ReductionContext.resolve_mapping(self._reduction_context))
            self._data.append(self._reduction_context.removed)
        return self._serializer(self._data)


def _format_traffic_actor(obj, context: EnvisionDataFormatter):
    assert type(obj) is TrafficActorState
    context.add(obj.actor_id, "actor_id", op=Operation.REDUCE)
    context.add(obj.lane_id, "lane_id", op=Operation.DELTA | Operation.REDUCE)
    context.add(obj.position, "position", op=Operation.FLATTEN)
    context.add_primitive(obj.heading)
    context.add_primitive(obj.speed)
    context.add(obj.events, "events", op=Operation.DELTA)
    context.add(obj.score, "score")
    for lane in context.layer(obj.waypoint_paths):
        for waypoint in context.layer(lane):
            with context.layer():
                context.add(waypoint, "waypoint")
    for dp in context.layer(obj.driven_path):
        context.add(dp, "driven_path_point", op=Operation.FLATTEN)
    for l_point in context.layer(obj.point_cloud):
        context.add(l_point, "lidar_point", op=Operation.FLATTEN)
    for geo in context.layer(obj.mission_route_geometry):
        for route_point in context.layer(geo):
            context.add(route_point, "route_point", op=Operation.FLATTEN)
    assert type(obj.actor_type) is TrafficActorType
    context.add(obj.actor_type, "actor_type", op=Operation.ONCE)
    assert type(obj.vehicle_type) is VehicleType
    context.add(obj.vehicle_type, "vehicle_type", op=Operation.ONCE)


def _format_state(obj: State, context: EnvisionDataFormatter):
    assert type(obj) is State
    context.add(obj.frame_time, "frame_time")
    context.add(obj.scenario_id, "scenario_id")
    context.add(obj.scenario_name, "scenario_name", op=Operation.ONCE)
    for _id, t in context.layer(obj.traffic.items()):
        with context.layer():
            # context.add(_id, "agent_id", op=Operation.REDUCE)
            context.add(t, "traffic")
    # context.add(
    #     obj.bubbles,
    #     "bubbles",
    #     select=lambda bbl: (bbl.geometry, bbl.pose),
    #     alternate=lambda bbl: bbl.pose,
    #     op=Operation.DELTA_ALTERNATE,
    # )  # TODO: On delta use position+heading as alternative
    for bubble in context.layer(obj.bubbles):
        for p in context.layer(bubble):
            context.add(p, "bubble_point", op=Operation.FLATTEN)
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


def _format_waypoint(obj: Waypoint, context: EnvisionDataFormatter):
    t = type(obj)
    assert t is Waypoint
    context.add(obj.pos, "position", op=Operation.FLATTEN)
    context.add_primitive(float(obj.heading))
    context.add(obj.lane_id, "lane_id", op=Operation.REDUCE)
    context.add_primitive(obj.lane_width)
    context.add_primitive(obj.speed_limit)
    context.add_primitive(obj.lane_index)


def _format_list(l: Union[list, tuple], context: EnvisionDataFormatter):
    assert isinstance(l, (list, tuple))
    for e in context.layer(l):
        context.add(e, "")


_formatter_map[TrafficActorState] = _format_traffic_actor
_formatter_map[State] = _format_state
_formatter_map[VehicleType] = _format_vehicle_type
_formatter_map[TrafficActorType] = _format_traffic_actor_type
_formatter_map[Events] = _format_events
_formatter_map[Waypoint] = _format_waypoint
_sequence_formatter_map[list] = _format_list
_sequence_formatter_map[tuple] = _format_list
