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

import numpy as np

from envision.types import State, TrafficActorState, TrafficActorType, VehicleType
from smarts.core.events import Events
from smarts.core.road_map import Waypoint
from smarts.core.utils.file import unpack


@unique
class Operation(IntEnum):
    NONE = 0
    """No special operation. Value will be sent as is."""
    REDUCE = 1
    """Convert the value to an integer and centralize the value into a mapping. Useful for
     reoccuring values"""
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
    """Mappings between an object and its reduction to an ID."""

    def __init__(
        self,
        mapping: Optional[Dict[Hashable, int]] = None,
        removed: Optional[List[int]] = None,
    ) -> None:
        self.current_id = 0
        self._mapping = mapping or {}
        self.removed = removed or []

    @property
    def has_ids(self):
        """If this reducing tool has currently reduced objects."""
        return len(self._mapping) > 0

    def resolve_mapping(self: "ReductionContext"):
        """The mappings that the context contains"""
        return {k: v for _, (k, v) in self._mapping.items()}

    def resolve_value(self: "ReductionContext", value: Hashable) -> int:
        """Map the value to an ID."""
        cc = self.current_id
        reduce, _ = self._mapping.setdefault(hash(value), (cc, value))
        if self.current_id == reduce:
            self.current_id += 1
        return reduce


class EnvisionDataFormatter:
    """A formatter to put envision state into a reduced format."""

    def __init__(
        self,
        id,
        serializer: Callable[[list], Any] = lambda d: d,
        float_decimals: int = 3,
        bool_as_int: bool = False,
    ):
        # self.seen_objects = context.seen_objects if context else set()
        self.id: Any = id
        self._data: List[Any] = []
        self._reduction_context = ReductionContext()
        self._serializer = serializer
        self._float_decimals = float_decimals
        self._bool_as_int = bool_as_int

    def reset(self, reset_reduction_context: bool = True):
        """Reset the current context in preparation for new serialization."""
        self._data = []
        if reset_reduction_context:
            self._reduction_context = ReductionContext()

    def add_any(self, obj: Any):
        """Format the given object to the current layer."""
        if type(obj) in _primitives:
            self.add_primitive(obj)
        else:
            self.add(obj, None)

    def add_primitive(self, obj: Any):
        """Add the given object as is to the given layer. Will decompose known primitives."""
        # TODO prevent cycles
        # if isinstance(value, (Object)) and value in self.seen_objects:
        #     return

        f = _formatter_map.get(type(obj))
        if f:
            f(obj, self)
        else:
            if isinstance(obj, float):
                obj = round(obj, self._float_decimals)
            elif self._bool_as_int and isinstance(obj, (bool, np.bool_)):
                obj = int(obj)
            self._data.append(obj)

    def add(
        self,
        value: Any,
        id_: str,
        op: Operation = Operation.NONE,
        alternate: Callable[[Any], Any] = None,
    ):
        """Format the given object to the current layer. Specified operations are performed."""
        outval = value
        f = _sequence_formatter_map.get(type(value))
        if op & Operation.REDUCE:
            outval = self._reduction_context.resolve_value(outval)
        if op & Operation.FLATTEN:
            outval = unpack(outval)
            if not isinstance(outval, (Sequence, np.ndarray)):
                assert False, "Must use flatten with Sequence or dataclass"
            for e in outval:
                self.add_primitive(e)
        elif f:
            f(value, self)
        else:
            self.add_primitive(outval)

    class DataFormatterLayer(ContextManager, Iterator):
        """A formatting layer that maps into the above layer of the current data formatter."""

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
        """Create a new layer which maps into a section of the above layer."""
        return self.DataFormatterLayer(self, iterable, op)

    def resolve(self) -> List:
        """Resolve all layers and mappings into the final data object."""
        if self._reduction_context.has_ids:
            self._data.append(self._reduction_context.resolve_mapping())
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
    # TODO: On delta use position+heading as alternative
    for bubble in context.layer(obj.bubbles):
        for p in context.layer(bubble):
            context.add(p, "bubble_point", op=Operation.FLATTEN)


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
