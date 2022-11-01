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
    NamedTuple,
    Optional,
    Sequence,
    Type,
    Union,
)

import numpy as np

from envision.types import (
    SignalLightState,
    SignalState,
    State,
    TrafficActorState,
    TrafficActorType,
    VehicleType,
)
from smarts.core.events import Events
from smarts.core.road_map import Waypoint
from smarts.core.utils.file import unpack


@unique
class Operation(IntEnum):
    """Formatting operations that should be performed on an object or layer."""

    NONE = 0
    """No special operation. Value will be sent as is."""
    REDUCE = 1
    """Send value only if it has changed."""
    FLATTEN = 4
    """Convert value from list or dataclass to higher hierachy."""


_formatter_map: Dict[Type, Callable[[Any, "EnvisionDataFormatter"], None]] = {}
_sequence_formatter_map: Dict[Type, Callable[[Any, "EnvisionDataFormatter"], None]] = {}

_primitives = {int, float, str, VehicleType, TrafficActorType}


class ReductionContext:
    """Mappings between an object and its reduction to an ID."""

    def __init__(
        self,
        mapping: Optional[Dict[Hashable, int]] = None,
        removed: Optional[List[int]] = None,
        enabled: bool = True,
    ) -> None:
        self._current_id = 0
        self._mapping = mapping or {}
        self.removed = removed or []
        self._enabled = enabled

    def reset(self):
        """Returns this context back to blank."""
        self._current_id = 0
        self._mapping = {}
        self.removed = []

    @property
    def enabled(self):
        """If this reduction context is enabled(else it is passthrough.)"""
        return self._enabled

    @property
    def has_ids(self):
        """If this reducing tool has currently reduced objects."""
        return len(self._mapping) > 0

    def resolve_mapping(self: "ReductionContext"):
        """The mappings that the context contains"""
        return {k: v for _, (k, v) in self._mapping.items()}

    def resolve_value(
        self: "ReductionContext", value: Hashable
    ) -> Union[int, Hashable]:
        """Map the value to an ID."""
        if not self._enabled:
            return value
        cc = self._current_id
        reduce, _ = self._mapping.setdefault(hash(value), (cc, value))
        if self._current_id == reduce:
            self._current_id += 1
        return reduce


class EnvisionDataFormatterArgs(NamedTuple):
    """Data formatter configurations."""

    id: Optional[str]
    serializer: Callable[[list], Any] = lambda d: d
    float_decimals: int = 2
    bool_as_int: bool = True
    enable_reduction: bool = True


class EnvisionDataFormatter:
    """A formatter to put envision state into a reduced format."""

    def __init__(
        self,
        formatter_params: EnvisionDataFormatterArgs,
    ):
        # self.seen_objects = context.seen_objects if context else set()
        self.id: Any = id
        self._data: List[Any] = []
        self._reduction_context = ReductionContext(
            enabled=formatter_params.enable_reduction
        )
        self._serializer = formatter_params.serializer
        self._float_decimals = formatter_params.float_decimals
        self._bool_as_int = formatter_params.bool_as_int

    def reset(self, reset_reduction_context: bool = True):
        """Reset the current context in preparation for new serialization."""
        self._data = []
        if reset_reduction_context:
            self._reduction_context.reset()

    def add_any(self, obj: Any):
        """Format the given object to the current layer."""
        if type(obj) in _primitives:
            self.add_primitive(obj)
        else:
            self.add(obj)

    def add_primitive(self, obj: Any):
        """Add the given object as is to the given layer. Will decompose known primitives."""
        # TODO prevent cycles
        # if isinstance(ref, (Object)) and ref in self.seen_objects:
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
        op: Operation = Operation.NONE,
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
            data_formatter: "EnvisionDataFormatter",
            iterable: Optional[Iterable],
            op: Operation,
        ) -> None:
            super().__init__()
            self._data_formatter = data_formatter
            self._upper_layer_data = data_formatter._data
            self._operation = op

            def empty_gen():
                return
                yield

            self._iterable: Generator[Any, None, None] = (
                (v for v in iterable) if iterable else empty_gen()
            )

        def __enter__(self):
            super().__enter__()
            self._data_formatter._data = []
            return self._iterable

        def __exit__(
            self,
            __exc_type: Optional[Type[BaseException]],
            __exc_value: Optional[BaseException],
            __traceback: Optional[TracebackType],
        ) -> Optional[bool]:
            d = self._data_formatter._data
            self._data_formatter._data = self._upper_layer_data
            self._data_formatter.add(d, op=self._operation)
            return super().__exit__(__exc_type, __exc_value, __traceback)

        def __iter__(self) -> Iterator[Any]:
            super().__iter__()
            self._data_formatter._data = []
            return self

        def __next__(self) -> Any:
            try:
                n = next(self._iterable)
                return n
            except StopIteration:
                d = self._data_formatter._data
                self._data_formatter._data = self._upper_layer_data
                self._data_formatter.add_primitive(d)
                raise

    def layer(
        self, iterable: Optional[Iterable] = None, op: Operation = Operation.NONE
    ):
        """Create a new layer which maps into a section of the above layer."""
        return self.DataFormatterLayer(self, iterable, op)

    def resolve(self) -> List:
        """Resolve all layers and mappings into the final data object."""
        if self._reduction_context.has_ids:
            self._data.append(self._reduction_context.resolve_mapping())
            self._data.append(self._reduction_context.removed)
        return self._serializer(self._data)


def _format_traffic_actor(obj, data_formatter: EnvisionDataFormatter):
    assert type(obj) is TrafficActorState
    data_formatter.add(obj.actor_id, op=Operation.REDUCE)
    data_formatter.add(obj.lane_id, op=Operation.REDUCE)
    data_formatter.add(obj.position, op=Operation.FLATTEN)
    data_formatter.add_primitive(obj.heading)
    data_formatter.add_primitive(obj.speed)
    data_formatter.add(obj.events)
    for lane in data_formatter.layer(obj.waypoint_paths):
        for waypoint in data_formatter.layer(lane):
            with data_formatter.layer():
                data_formatter.add(waypoint)
    for dp in data_formatter.layer(obj.driven_path):
        data_formatter.add(dp, op=Operation.FLATTEN)
    for l_point in data_formatter.layer(obj.point_cloud):
        data_formatter.add(l_point, op=Operation.FLATTEN)
    for geo in data_formatter.layer(obj.mission_route_geometry):
        for route_point in data_formatter.layer(geo):
            data_formatter.add(route_point, op=Operation.FLATTEN)
    assert type(obj.actor_type) is TrafficActorType
    data_formatter.add(obj.actor_type)
    assert type(obj.vehicle_type) is VehicleType
    data_formatter.add(obj.vehicle_type)


def _format_state(obj: State, data_formatter: EnvisionDataFormatter):
    assert type(obj) is State
    data_formatter.add(obj.frame_time)
    data_formatter.add(obj.scenario_id)
    data_formatter.add(obj.scenario_name)
    for t in data_formatter.layer(obj.traffic.values()):
        with data_formatter.layer():
            # context.add(_id, op=Operation.REDUCE)
            data_formatter.add(t)
    for s in data_formatter.layer(obj.signals.values()):
        with data_formatter.layer():
            data_formatter.add(s)
    # TODO: On delta use position+heading as alternative
    for bubble in data_formatter.layer(obj.bubbles):
        for p in data_formatter.layer(bubble):
            data_formatter.add(p, op=Operation.FLATTEN)
    for id_, score in data_formatter.layer(obj.scores.items()):
        with data_formatter.layer():
            data_formatter.add(id_, op=Operation.REDUCE)
            data_formatter.add(score)
    for id_ in data_formatter.layer(obj.ego_agent_ids):
        data_formatter.add(id_, op=Operation.REDUCE)


def _format_vehicle_type(obj: VehicleType, data_formatter: EnvisionDataFormatter):
    t = type(obj)
    assert t is VehicleType
    mapping = {
        VehicleType.Bus: 0,
        VehicleType.Coach: 1,
        VehicleType.Truck: 2,
        VehicleType.Trailer: 3,
        VehicleType.Car: 4,
        VehicleType.Motorcycle: 5,
        VehicleType.Pedestrian: 6,
    }
    data_formatter.add_primitive(mapping[obj])


def _format_traffic_actor_type(
    obj: TrafficActorType, data_formatter: EnvisionDataFormatter
):
    t = type(obj)
    assert t is TrafficActorType
    mapping = {
        TrafficActorType.SocialVehicle: 0,
        TrafficActorType.SocialAgent: 1,
        TrafficActorType.Agent: 2,
    }
    data_formatter.add_primitive(mapping[obj])


def _format_signal_light_state(
    obj: SignalLightState, data_formatter: EnvisionDataFormatter
):
    t = type(obj)
    assert t is SignalLightState
    mapping = {
        SignalLightState.Unknown: 0,
        SignalLightState.Off: 0,
        SignalLightState.Stop: 1,
        SignalLightState.Caution: 2,
        SignalLightState.Go: 3,
    }
    data_formatter.add_primitive(mapping[obj])


def _format_signal_state(obj: SignalState, data_formatter: EnvisionDataFormatter):
    assert type(obj) is SignalState
    data_formatter.add(obj.signal_id, op=Operation.REDUCE)
    assert type(obj.signal_light_state) is SignalLightState
    data_formatter.add(obj.signal_light_state)


def _format_events(obj: Events, data_formatter: EnvisionDataFormatter):
    t = type(obj)
    assert t is Events
    data_formatter.add_primitive(tuple(obj))


def _format_waypoint(obj: Waypoint, data_formatter: EnvisionDataFormatter):
    t = type(obj)
    assert t is Waypoint
    data_formatter.add(obj.pos, op=Operation.FLATTEN)
    data_formatter.add_primitive(float(obj.heading))
    data_formatter.add(obj.lane_id, op=Operation.REDUCE)
    data_formatter.add_primitive(obj.lane_width)
    data_formatter.add_primitive(obj.speed_limit)
    data_formatter.add_primitive(obj.lane_index)
    data_formatter.add_primitive(obj.lane_offset)


def _format_list(l: Union[list, tuple], data_formatter: EnvisionDataFormatter):
    assert isinstance(l, (list, tuple))
    for e in data_formatter.layer(l):
        data_formatter.add(e)


_formatter_map[TrafficActorState] = _format_traffic_actor
_formatter_map[State] = _format_state
_formatter_map[SignalState] = _format_signal_state
_formatter_map[SignalLightState] = _format_signal_light_state
_formatter_map[VehicleType] = _format_vehicle_type
_formatter_map[TrafficActorType] = _format_traffic_actor_type
_formatter_map[Events] = _format_events
_formatter_map[Waypoint] = _format_waypoint
_sequence_formatter_map[list] = _format_list
_sequence_formatter_map[tuple] = _format_list
